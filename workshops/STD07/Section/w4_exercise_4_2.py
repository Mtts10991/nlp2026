import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import numpy as np
from contextlib import redirect_stdout

# ===== Import จาก w4 source — suppress module-level prints =====
with redirect_stdout(io.StringIO()):
    from w4_bert_finetune import vocab_expansion_demo, MockTokenize

# ===== โค้ดบังคับตามโจทย์ =====
print("=" * 70)
print("  รัน vocab_expansion_demo() จาก source")
print("=" * 70)
vocab_expansion_demo()

print("\n" + "=" * 70)
print("  ทดลองเพิ่ม legal terms ของตัวเอง")
print("=" * 70)
MY_LEGAL_TERMS = ["อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ"]
base_vocab = 5000
new_vocab = {t: base_vocab + i for i, t in enumerate(MY_LEGAL_TERMS)}
print(new_vocab)

# ================================================================
# (oo) อธิบาย Catastrophic Forgetting — เกิดยังไงเมื่อเพิ่มคำใหม่แล้ว fine-tune ทันที
# W 4.2 : OO อธิบาย Catastrophic Forgetting — เกิดขึ้นได้อย่างไรเมื่อเพิ่มคำใหม่แล้ว fine-tune ทันที
# ================================================================
print("\n" + "=" * 70)
print("  (oo) Catastrophic Forgetting")
print("=" * 70)

print(f"""
  ▶ Catastrophic Forgetting คือ:
    "model ลืมความรู้เดิมที่ pre-train มา เมื่อ train งานใหม่"

  ▶ กลไกที่เกิดขึ้นเมื่อเพิ่มคำใหม่ + fine-tune ทันที:

    Step 1: เพิ่มคำใหม่เข้า vocab
      'อนุสิทธิบัตร' → token id = 5000  (ใหม่)
      embedding[5000] = random ❗ (ยังไม่มีความหมาย)

    Step 2: Fine-tune ทุก layer พร้อมกัน
      Forward pass:
        คำใหม่ → random embedding → ผ่าน 12 layers ของ BERT
        → output ผิดมาก (เพราะ random)
      Backward pass:
        gradient ใหญ่ (เพราะ loss สูง)
        → gradient ไหลกลับไปทุก weight ของ BERT
        → weights ที่ pre-trained มาดี ๆ ถูก "ดึง" ไปทาง random

    Step 3: ผลที่ตามมา
      • Embedding ของคำเดิม (เช่น 'สิทธิ', 'ละเมิด') ถูกแก้ไขจน
        ไม่ใกล้ความหมายเดิม → semantic เสียหาย
      • Attention pattern ที่เรียนรู้มา (e.g. คำคุณศัพท์ขยายคำนาม)
        ถูก override
      • Accuracy บน task ทั่วไป (NLI, QA) ตกฮวบ
        แม้ accuracy บน task ใหม่จะดี (overfitting + forgetting)

  ▶ analogy:
    เหมือนเรียนภาษาญี่ปุ่นเข้มข้นเป็นเวลา 1 เดือน → จะหลงลืมภาษาเกาหลี
    ที่เคยเรียนมาเก่า ๆ (interference learning ใน cognitive science)

  ▶ วิธีแก้ — Warm-up (ดูข้อ pp)
""")

# ================================================================
# (pp) Warm-up 3 ขั้นตอนคืออะไร — ทำไมขั้น 2 ต้องใช้ LR=2e-5 ไม่ใช่ 1e-3
# W 4.2 : PP Warm-up 3 ขั้นตอนคืออะไร — ทำไมขั้น 2 ต้องใช้ LR=2e-5 ไม่ใช่ 1e-3
# ================================================================
print("=" * 70)
print("  (pp) Warm-up 3 ขั้นตอน")
print("=" * 70)

print(f"""
  ▶ Warm-up Strategy (จาก source's vocab_expansion_demo):

    ┌──────┬─────────────────────────────────┬──────────────────┐
    │ ขั้น │ การกระทำ                          │ Why                │
    ├──────┼─────────────────────────────────┼──────────────────┤
    │  1   │ Freeze ทุก layer ยกเว้น embedding  │ ให้คำใหม่หา        │
    │      │ Train แค่ embedding 2-3 epochs    │ embedding ที่     │
    │      │                                   │ "สอดคล้อง" กับ      │
    │      │                                   │ context ก่อน         │
    ├──────┼─────────────────────────────────┼──────────────────┤
    │  2   │ Unfreeze ทุก layer                │ ปรับทุกส่วนพร้อมกัน │
    │      │ Train ด้วย LR ต่ำ (2e-5)          │ แต่อย่างค่อย ๆ      │
    │      │                                   │ ไม่กระชาก           │
    ├──────┼─────────────────────────────────┼──────────────────┤
    │  3   │ Fine-tune จนกว่า val_loss นิ่ง     │ Convergence final  │
    └──────┴─────────────────────────────────┴──────────────────┘

  ▶ ทำไมขั้น 2 ใช้ LR=2e-5 ไม่ใช่ 1e-3 (LR ต่ำกว่า 50 เท่า):

    1. Pre-trained weights = "ของมีค่า" — แตะนิดเดียวก็ได้:
       BERT ใช้เวลา pre-train หลายวันบน GPU 16 ตัว
       weights อยู่ใน "good neighborhood" ของ loss landscape แล้ว
       LR สูง = step ใหญ่ = อาจกระโดดออกจาก neighborhood นั้น

    2. ป้องกัน Catastrophic Forgetting:
       LR=1e-3 → gradient × 1e-3 (step ใหญ่) → weights เปลี่ยนเยอะ
       LR=2e-5 → gradient × 2e-5 (step เล็ก 50 เท่า) → ค่อย ๆ ปรับ
       → เก็บความรู้เดิมไว้ + ปรับทิศทางเล็กน้อยให้เข้า task ใหม่

    3. Standard rule จาก BERT paper:
       Devlin et al. แนะนำ LR ∈ [2e-5, 5e-5] สำหรับ fine-tuning
       LR > 5e-5 → unstable, อาจ diverge
       LR < 2e-5 → train ช้าเกินไป

    4. Loss landscape geometry:
       Pre-trained model อยู่ใกล้ minimum แล้ว
       LR สูง = อาจผ่านเลย minimum → loss ขึ้น
       LR ต่ำ = ปรับให้เข้า minimum นั้นจริง ๆ

  ▶ Trade-off ของ Warm-up:
    ✅ ป้องกัน Catastrophic Forgetting
    ✅ Convergence stable
    ❌ ใช้เวลามากกว่า (3 phases vs 1 phase end-to-end)
    ❌ ต้องจัดการ optimizer state เปลี่ยนระหว่าง phase
""")

# ================================================================
# (qq) cosine similarity ก่อน ≈ 0.38 และหลัง ≈ 0.96 หมายความว่าอะไร
# W 4.2 : QQ cosine similarity ก่อน warm-up ≈ 0.38 และหลัง ≈ 0.96 หมายความว่าอะไร — อธิบายใน vector space
# ================================================================
print("=" * 70)
print("  (qq) Cosine Similarity ก่อน/หลัง warm-up")
print("=" * 70)

# Replicate การคำนวณจาก source's vocab_expansion_demo
np.random.seed(42)
d_model = 8
old_emb = np.array([0.82, -0.34, 0.56, 0.91, -0.12, 0.67, -0.45, 0.23])
new_before = np.random.randn(d_model) * 0.02
new_after = old_emb * 0.6 + np.random.randn(d_model) * 0.1

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_before = cosine(old_emb, new_before)
cos_after = cosine(old_emb, new_after)

print(f"""
  ▶ ตัวเลขจริงจาก vocab_expansion_demo:
    cosine(old, new_before warm-up) = {cos_before:+.3f}
    cosine(old, new_after warm-up)  = {cos_after:+.3f}

  ▶ ความหมายของ cosine similarity:
    cosine(A, B) = (A · B) / (|A| × |B|)
    = cos(มุมระหว่าง A กับ B ใน vector space)

    ค่าอยู่ใน [-1, +1]:
       +1.0  → vector ชี้ไปทิศเดียวกัน  (semantic เหมือนกัน)
        0.0  → vector ตั้งฉากกัน          (semantic ไม่เกี่ยว)
       -1.0  → vector ชี้ตรงข้าม           (semantic ตรงข้าม)

  ▶ ตีความ:

    ก่อน warm-up: cosine ≈ 0.38
      → vector คำใหม่ ('อนุสิทธิบัตร') เกือบ "สุ่ม" ใน 8-D space
      → ไม่มีทิศทางที่สอดคล้องกับคำเดิม ('ละเมิด')
      → BERT มอง 2 คำนี้ "ไม่เกี่ยวข้องกัน"
      → semantic ของคำใหม่ = noise

    หลัง warm-up: cosine ≈ 0.96
      → vector คำใหม่ "ใกล้เคียง" คำเดิมมาก (cos ≈ 1)
      → embedding ถูก update ให้ชี้ไปทิศที่ "make sense" สำหรับ context
      → BERT มอง 2 คำนี้ "อยู่ใน semantic neighborhood เดียวกัน"

  ▶ Geometry ใน vector space:

    8-D space (ลดเหลือ 2-D เพื่อ visualize):

         old_emb ──────►  ('ละเมิด')
            ↑
            │  cos = 0.96
            │ (เกือบขนาน)
            │
         new_after ─►  ('อนุสิทธิบัตร' หลัง warm-up)
                                       ▲
                                       │
                                       │ cos = 0.38
                                       │ (เกือบตั้งฉาก)
                                       │
              new_before •  ('อนุสิทธิบัตร' ก่อน warm-up — random)

  ▶ เพราะอะไรถึงสำคัญสำหรับ BERT:

    BERT ทำ self-attention ด้วย dot-product (เกี่ยวข้องกับ cosine)
    คำที่ embedding ใกล้กัน → attend หากันบ่อย → ส่งต่อ info
    คำใหม่ที่ random → ไม่ attend ใคร → เป็น "เกาะ" ที่ไม่มีใครคุยด้วย
    Warm-up = เชื่อม "เกาะ" ใหม่นี้เข้ากับ network เดิม
""")

# ================================================================
# (rr) เสนอคำศัพท์กฎหมาย IP ที่ควรเพิ่มเข้า vocab อีกอย่างน้อย 5 คำ
# W 4.2 : RR เสนอคำศัพท์กฎหมาย IP ที่ควรเพิ่มเข้า vocab อีกอย่างน้อย 5 คำ พร้อมให้เหตุผล
# ================================================================
print("=" * 70)
print("  (rr) คำศัพท์กฎหมาย IP ที่ควรเพิ่มเข้า vocab")
print("=" * 70)

# คำที่เสนอเพิ่ม + เหตุผล (อ้างอิงจาก dataset และ legal domain)
SUGGESTED_TERMS = [
    {
        "term": "เครื่องหมายการค้า",
        "category": "Trademark",
        "reason": "หมวด IP สำคัญที่ dataset ปัจจุบันยังไม่ครอบคลุม (มีแค่ patent + copyright) — ถ้าจะขยายเป็น 4-class ในอนาคต จะต้องการ embedding ที่แม่น",
        "freq_in_corpus": "0 (ยังไม่มีใน dataset แต่จำเป็นสำหรับ class ในอนาคต)"
    },
    {
        "term": "ผู้ทรงสิทธิ",
        "category": "Procedural",
        "reason": "บุคคลในคดี IP ทุกประเภท — เป็น stakeholder หลัก (เจ้าของสิทธิ์ที่ฟ้องคดี)",
        "freq_in_corpus": "ปรากฏใน LEGAL_GLOSSARY ของ W1"
    },
    {
        "term": "พระราชบัญญัติ",
        "category": "Legal Document",
        "reason": "ชื่อกฎหมายเต็มที่อ้างใน IP cases ทุกคดี (พ.ร.บ.สิทธิบัตร, พ.ร.บ.ลิขสิทธิ์)",
        "freq_in_corpus": "ปรากฏใน LEGAL_GLOSSARY"
    },
    {
        "term": "ศาลทรัพย์สินทางปัญญา",
        "category": "Court",
        "reason": "ศาลพิเศษที่พิจารณาคดี IP เฉพาะ — entity ที่จำเป็นต้องระบุได้เพื่อ procedural classification",
        "freq_in_corpus": "ปรากฏใน LEGAL_GLOSSARY (PROCEDURE)"
    },
    {
        "term": "ระวางโทษ",
        "category": "Penalty",
        "reason": "Phrase สำคัญที่ตามด้วยโทษจำคุก/ปรับ — sensor trigger สำหรับ Physics Gate (W17)",
        "freq_in_corpus": "ปรากฏใน LEGAL_GLOSSARY (ENFORCEMENT)"
    },
    {
        "term": "การประดิษฐ์ขึ้นใหม่",
        "category": "Patent",
        "reason": "เกณฑ์หลักของการได้สิทธิบัตร (novelty requirement) — distinctive marker ของคดี patent",
        "freq_in_corpus": "ปรากฏใน LEGAL_COMPOUNDS ของ W1"
    },
    {
        "term": "งานสร้างสรรค์",
        "category": "Copyright",
        "reason": "Subject ของลิขสิทธิ์ — distinguish copyright case จาก patent",
        "freq_in_corpus": "ปรากฏใน LEGAL_GLOSSARY (COPYRIGHT)"
    },
]

print(f"\n  เสนอ {len(SUGGESTED_TERMS)} คำใหม่ (เกินขั้นต่ำ 5 คำ):\n")
print(f"  {'#':<3} {'Term':<30} {'Category':<15} {'พบใน corpus?'}")
print(f"  {'-'*3} {'-'*30} {'-'*15} {'-'*40}")
for i, item in enumerate(SUGGESTED_TERMS, 1):
    print(f"  {i:<3} {item['term']:<30} {item['category']:<15} {item['freq_in_corpus']}")

print(f"\n  ▶ เหตุผลรายตัว:")
for i, item in enumerate(SUGGESTED_TERMS, 1):
    print(f"\n  {i}. '{item['term']}' [{item['category']}]")
    print(f"     {item['reason']}")

print(f"""

  ▶ Methodology การเลือก:
    1. ดู LEGAL_GLOSSARY ใน W1 (ครอบคลุม 5 หมวด: PATENT, COPYRIGHT,
       TRADEMARK, ENFORCEMENT, PROCEDURE)
    2. เลือกคำที่ปรากฏใน corpus หรือเกี่ยวกับ legal proceedings
    3. ครอบคลุมหลายหมวด (ไม่กระจุกอยู่ patent อย่างเดียว)
    4. เลือก compound words ที่ tokenizer มักตัดผิด

  ▶ Extended new_vocab:
""")

EXTENDED = MY_LEGAL_TERMS + [item["term"] for item in SUGGESTED_TERMS]
extended_vocab = {t: base_vocab + i for i, t in enumerate(EXTENDED)}
for t, idx in extended_vocab.items():
    new_marker = " ← เพิ่มใหม่" if t not in MY_LEGAL_TERMS else ""
    print(f"    '{t}' → id {idx}{new_marker}")


"""

======================================================================
  รัน vocab_expansion_demo() จาก source
======================================================================
==========================================================
  STEP 1: ก่อน expansion — คำใหม่ถูกตัดเป็นชิ้น
==========================================================

  ข้อความ    : 'สิทธิบัตรการประดิษฐ์'
  input_ids  : [1, 3726, 3736, 3707, 3708, 3736, 3710, 3733, 3705, 3719, 3685, 3734, 3719, 3711, 3719, 3732, 3704, 3736, 3725, 3700, 3760, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  mask       : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  real tokens: 22  PAD: 10

  แยกส่วน:
    [CLS] = 1
    tokens= [3726, 3736, 3707, 3708, 3736, 3710, 3733, 3705, 3719, 3685, 3734, 3719, 3711, 3719, 3732, 3704, 3736, 3725, 3700, 3760]
    [SEP] = 2
    [PAD] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

==========================================================
  STEP 2: เพิ่มคำใหม่เข้า vocab (Vocabulary Expansion)
==========================================================

  vocab เดิม : 5000 คำ
  เพิ่มคำใหม่: 5 คำ
  vocab ใหม่ : 5005 คำ

    'ภูมิปัญญาท้องถิ่น' → id 5000  (token ใหม่ weight = random ❗)
    'สิทธิการประดิษฐ์' → id 5001  (token ใหม่ weight = random ❗)
    'อนุสิทธิบัตร' → id 5002  (token ใหม่ weight = random ❗)
    'ทรัพย์สินทางปัญญา' → id 5003  (token ใหม่ weight = random ❗)
    'การละเมิดสิทธิ' → id 5004  (token ใหม่ weight = random ❗)

==========================================================
  STEP 3: ทำไมต้อง Warm-up ก่อน Fine-tune
==========================================================

  ปัญหา:
    คำเดิม  → weights ผ่าน pre-train มาแล้ว (มีความหมาย)
    คำใหม่  → weights = random ❗ (ยังไม่มีความหมาย)

  ถ้า fine-tune ทุก layer พร้อมกันเลย:
    gradient จากคำใหม่ (random) จะรบกวน weights เดิม
    → โมเดลลืมสิ่งที่เรียนมา = Catastrophic Forgetting ❌

  วิธีแก้ — Warm-up 3 ขั้นตอน:

    ขั้น 1 │ Freeze ทุก layer ยกเว้น embedding
           │ train แค่ embedding 2-3 epochs
           │ → คำใหม่เริ่มมีความหมาย

    ขั้น 2 │ Unfreeze ทุก layer
           │ train ด้วย LR ต่ำ (2e-5)
           │ → ปรับ weights ทั้งหมดพร้อมกัน

    ขั้น 3 │ Fine-tune จนกว่า val_loss นิ่ง
           │ → โมเดลพร้อมใช้งาน ✅
    
==========================================================
  STEP 4: จำลอง embedding weight ก่อน/หลัง warm-up
==========================================================

  คำเดิม  'ละเมิด'       : [ 0.82 -0.34  0.56  0.91 -0.12  0.67 -0.45  0.23]
  คำใหม่ ก่อน warm-up   : [ 0.01 -0.    0.01  0.03 -0.   -0.    0.03  0.02]  ← random
  คำใหม่ หลัง warm-up   : [ 0.45 -0.15  0.29  0.5  -0.05  0.21 -0.44  0.08]   ← มีทิศทางแล้ว

  cosine similarity กับ 'ละเมิด':
    ก่อน warm-up : 0.380  (ไม่เกี่ยวกัน)
    หลัง warm-up : 0.958   (ใกล้เคียงกัน)

======================================================================
  ทดลองเพิ่ม legal terms ของตัวเอง
======================================================================
{'อนุสิทธิบัตร': 5000, 'ทรัพย์สินทางปัญญา': 5001, 'การละเมิดสิทธิ': 5002}

======================================================================
  (oo) Catastrophic Forgetting
======================================================================

  ▶ Catastrophic Forgetting คือ:
    "model ลืมความรู้เดิมที่ pre-train มา เมื่อ train งานใหม่"

  ▶ กลไกที่เกิดขึ้นเมื่อเพิ่มคำใหม่ + fine-tune ทันที:

    Step 1: เพิ่มคำใหม่เข้า vocab
      'อนุสิทธิบัตร' → token id = 5000  (ใหม่)
      embedding[5000] = random ❗ (ยังไม่มีความหมาย)

    Step 2: Fine-tune ทุก layer พร้อมกัน
      Forward pass:
        คำใหม่ → random embedding → ผ่าน 12 layers ของ BERT
        → output ผิดมาก (เพราะ random)
      Backward pass:
        gradient ใหญ่ (เพราะ loss สูง)
        → gradient ไหลกลับไปทุก weight ของ BERT
        → weights ที่ pre-trained มาดี ๆ ถูก "ดึง" ไปทาง random

    Step 3: ผลที่ตามมา
      • Embedding ของคำเดิม (เช่น 'สิทธิ', 'ละเมิด') ถูกแก้ไขจน
        ไม่ใกล้ความหมายเดิม → semantic เสียหาย
      • Attention pattern ที่เรียนรู้มา (e.g. คำคุณศัพท์ขยายคำนาม)
        ถูก override
      • Accuracy บน task ทั่วไป (NLI, QA) ตกฮวบ
        แม้ accuracy บน task ใหม่จะดี (overfitting + forgetting)

  ▶ analogy:
    เหมือนเรียนภาษาญี่ปุ่นเข้มข้นเป็นเวลา 1 เดือน → จะหลงลืมภาษาเกาหลี
    ที่เคยเรียนมาเก่า ๆ (interference learning ใน cognitive science)

  ▶ วิธีแก้ — Warm-up (ดูข้อ pp)

======================================================================
  (pp) Warm-up 3 ขั้นตอน
======================================================================

  ▶ Warm-up Strategy (จาก source's vocab_expansion_demo):

    ┌──────┬─────────────────────────────────┬──────────────────┐
    │ ขั้น │ การกระทำ                          │ Why                │
    ├──────┼─────────────────────────────────┼──────────────────┤
    │  1   │ Freeze ทุก layer ยกเว้น embedding  │ ให้คำใหม่หา        │
    │      │ Train แค่ embedding 2-3 epochs    │ embedding ที่     │
    │      │                                   │ "สอดคล้อง" กับ      │
    │      │                                   │ context ก่อน         │
    ├──────┼─────────────────────────────────┼──────────────────┤
    │  2   │ Unfreeze ทุก layer                │ ปรับทุกส่วนพร้อมกัน │
    │      │ Train ด้วย LR ต่ำ (2e-5)          │ แต่อย่างค่อย ๆ      │
    │      │                                   │ ไม่กระชาก           │
    ├──────┼─────────────────────────────────┼──────────────────┤
    │  3   │ Fine-tune จนกว่า val_loss นิ่ง     │ Convergence final  │
    └──────┴─────────────────────────────────┴──────────────────┘

  ▶ ทำไมขั้น 2 ใช้ LR=2e-5 ไม่ใช่ 1e-3 (LR ต่ำกว่า 50 เท่า):

    1. Pre-trained weights = "ของมีค่า" — แตะนิดเดียวก็ได้:
       BERT ใช้เวลา pre-train หลายวันบน GPU 16 ตัว
       weights อยู่ใน "good neighborhood" ของ loss landscape แล้ว
       LR สูง = step ใหญ่ = อาจกระโดดออกจาก neighborhood นั้น

    2. ป้องกัน Catastrophic Forgetting:
       LR=1e-3 → gradient × 1e-3 (step ใหญ่) → weights เปลี่ยนเยอะ
       LR=2e-5 → gradient × 2e-5 (step เล็ก 50 เท่า) → ค่อย ๆ ปรับ
       → เก็บความรู้เดิมไว้ + ปรับทิศทางเล็กน้อยให้เข้า task ใหม่

    3. Standard rule จาก BERT paper:
       Devlin et al. แนะนำ LR ∈ [2e-5, 5e-5] สำหรับ fine-tuning
       LR > 5e-5 → unstable, อาจ diverge
       LR < 2e-5 → train ช้าเกินไป

    4. Loss landscape geometry:
       Pre-trained model อยู่ใกล้ minimum แล้ว
       LR สูง = อาจผ่านเลย minimum → loss ขึ้น
       LR ต่ำ = ปรับให้เข้า minimum นั้นจริง ๆ

  ▶ Trade-off ของ Warm-up:
    ✅ ป้องกัน Catastrophic Forgetting
    ✅ Convergence stable
    ❌ ใช้เวลามากกว่า (3 phases vs 1 phase end-to-end)
    ❌ ต้องจัดการ optimizer state เปลี่ยนระหว่าง phase

======================================================================
  (qq) Cosine Similarity ก่อน/หลัง warm-up
======================================================================

  ▶ ตัวเลขจริงจาก vocab_expansion_demo:
    cosine(old, new_before warm-up) = +0.380
    cosine(old, new_after warm-up)  = +0.958

  ▶ ความหมายของ cosine similarity:
    cosine(A, B) = (A · B) / (|A| × |B|)
    = cos(มุมระหว่าง A กับ B ใน vector space)

    ค่าอยู่ใน [-1, +1]:
       +1.0  → vector ชี้ไปทิศเดียวกัน  (semantic เหมือนกัน)
        0.0  → vector ตั้งฉากกัน          (semantic ไม่เกี่ยว)
       -1.0  → vector ชี้ตรงข้าม           (semantic ตรงข้าม)

  ▶ ตีความ:

    ก่อน warm-up: cosine ≈ 0.38
      → vector คำใหม่ ('อนุสิทธิบัตร') เกือบ "สุ่ม" ใน 8-D space
      → ไม่มีทิศทางที่สอดคล้องกับคำเดิม ('ละเมิด')
      → BERT มอง 2 คำนี้ "ไม่เกี่ยวข้องกัน"
      → semantic ของคำใหม่ = noise

    หลัง warm-up: cosine ≈ 0.96
      → vector คำใหม่ "ใกล้เคียง" คำเดิมมาก (cos ≈ 1)
      → embedding ถูก update ให้ชี้ไปทิศที่ "make sense" สำหรับ context
      → BERT มอง 2 คำนี้ "อยู่ใน semantic neighborhood เดียวกัน"

  ▶ Geometry ใน vector space:

    8-D space (ลดเหลือ 2-D เพื่อ visualize):

         old_emb ──────►  ('ละเมิด')
            ↑
            │  cos = 0.96
            │ (เกือบขนาน)
            │
         new_after ─►  ('อนุสิทธิบัตร' หลัง warm-up)
                                       ▲
                                       │
                                       │ cos = 0.38
                                       │ (เกือบตั้งฉาก)
                                       │
              new_before •  ('อนุสิทธิบัตร' ก่อน warm-up — random)

  ▶ เพราะอะไรถึงสำคัญสำหรับ BERT:

    BERT ทำ self-attention ด้วย dot-product (เกี่ยวข้องกับ cosine)
    คำที่ embedding ใกล้กัน → attend หากันบ่อย → ส่งต่อ info
    คำใหม่ที่ random → ไม่ attend ใคร → เป็น "เกาะ" ที่ไม่มีใครคุยด้วย
    Warm-up = เชื่อม "เกาะ" ใหม่นี้เข้ากับ network เดิม

======================================================================
  (rr) คำศัพท์กฎหมาย IP ที่ควรเพิ่มเข้า vocab
======================================================================

  เสนอ 7 คำใหม่ (เกินขั้นต่ำ 5 คำ):

  #   Term                           Category        พบใน corpus?
  --- ------------------------------ --------------- ----------------------------------------
  1   เครื่องหมายการค้า              Trademark       0 (ยังไม่มีใน dataset แต่จำเป็นสำหรับ class ในอนาคต)
  2   ผู้ทรงสิทธิ                    Procedural      ปรากฏใน LEGAL_GLOSSARY ของ W1
  3   พระราชบัญญัติ                  Legal Document  ปรากฏใน LEGAL_GLOSSARY
  4   ศาลทรัพย์สินทางปัญญา           Court           ปรากฏใน LEGAL_GLOSSARY (PROCEDURE)
  5   ระวางโทษ                       Penalty         ปรากฏใน LEGAL_GLOSSARY (ENFORCEMENT)
  6   การประดิษฐ์ขึ้นใหม่            Patent          ปรากฏใน LEGAL_COMPOUNDS ของ W1
  7   งานสร้างสรรค์                  Copyright       ปรากฏใน LEGAL_GLOSSARY (COPYRIGHT)

  ▶ เหตุผลรายตัว:

  1. 'เครื่องหมายการค้า' [Trademark]
     หมวด IP สำคัญที่ dataset ปัจจุบันยังไม่ครอบคลุม (มีแค่ patent + copyright) — ถ้าจะขยายเป็น 4-class ในอนาคต จะต้องการ embedding ที่แม่น

  2. 'ผู้ทรงสิทธิ' [Procedural]
     บุคคลในคดี IP ทุกประเภท — เป็น stakeholder หลัก (เจ้าของสิทธิ์ที่ฟ้องคดี)

  3. 'พระราชบัญญัติ' [Legal Document]
     ชื่อกฎหมายเต็มที่อ้างใน IP cases ทุกคดี (พ.ร.บ.สิทธิบัตร, พ.ร.บ.ลิขสิทธิ์)

  4. 'ศาลทรัพย์สินทางปัญญา' [Court]
     ศาลพิเศษที่พิจารณาคดี IP เฉพาะ — entity ที่จำเป็นต้องระบุได้เพื่อ procedural classification

  5. 'ระวางโทษ' [Penalty]
     Phrase สำคัญที่ตามด้วยโทษจำคุก/ปรับ — sensor trigger สำหรับ Physics Gate (W17)

  6. 'การประดิษฐ์ขึ้นใหม่' [Patent]
     เกณฑ์หลักของการได้สิทธิบัตร (novelty requirement) — distinctive marker ของคดี patent

  7. 'งานสร้างสรรค์' [Copyright]
     Subject ของลิขสิทธิ์ — distinguish copyright case จาก patent


  ▶ Methodology การเลือก:
    1. ดู LEGAL_GLOSSARY ใน W1 (ครอบคลุม 5 หมวด: PATENT, COPYRIGHT,
       TRADEMARK, ENFORCEMENT, PROCEDURE)
    2. เลือกคำที่ปรากฏใน corpus หรือเกี่ยวกับ legal proceedings
    3. ครอบคลุมหลายหมวด (ไม่กระจุกอยู่ patent อย่างเดียว)
    4. เลือก compound words ที่ tokenizer มักตัดผิด

  ▶ Extended new_vocab:

    'อนุสิทธิบัตร' → id 5000
    'ทรัพย์สินทางปัญญา' → id 5001
    'การละเมิดสิทธิ' → id 5002
    'เครื่องหมายการค้า' → id 5003 ← เพิ่มใหม่
    'ผู้ทรงสิทธิ' → id 5004 ← เพิ่มใหม่
    'พระราชบัญญัติ' → id 5005 ← เพิ่มใหม่
    'ศาลทรัพย์สินทางปัญญา' → id 5006 ← เพิ่มใหม่
    'ระวางโทษ' → id 5007 ← เพิ่มใหม่
    'การประดิษฐ์ขึ้นใหม่' → id 5008 ← เพิ่มใหม่
    'งานสร้างสรรค์' → id 5009 ← เพิ่มใหม่

"""
