import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
from contextlib import redirect_stdout

# ===== Import จาก w4 source — suppress module-level prints (show_model_comparison + get_device) =====
with redirect_stdout(io.StringIO()):
    from w4_bert_finetune import MockTokenize, MODEL_REGISTRY

# ===== โค้ดบังคับตามโจทย์ =====
tok = MockTokenize()
sentences = [
    "จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์",
    "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้อง",
    "ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์",
]

print("=" * 70)
print("  MockTokenize.show() บน 3 ประโยคจาก dataset")
print("=" * 70)
for s in sentences:
    tok.show(s, max_length=32)

# ================================================================
# (ll) อธิบาย input_ids structure — [CLS]=1, token_ids, [SEP]=2, [PAD]=0
# W 4.1 : LL อธิบาย input_ids structure: [CLS]=1, token_ids, [SEP]=2, [PAD]=0 — แต่ละส่วนมีบทบาทอะไรใน BERT
# ================================================================
print("\n" + "=" * 70)
print("  (ll) input_ids structure — บทบาทใน BERT")
print("=" * 70)

print(f"""
  ▶ Pattern: [CLS] + token_ids + [SEP] + [PAD]...

    [CLS] = 1   "Classification token" — token พิเศษที่อยู่ตำแหน่งแรกเสมอ
    [SEP] = 2   "Separator" — ปิดท้ายประโยค (หรือคั่นระหว่าง 2 ประโยค)
    [PAD] = 0   "Padding" — เติมให้ทุก sequence ยาวเท่ากัน
    token = {{อื่น ๆ}}  vocab id ของแต่ละ token

  ▶ บทบาทใน BERT:

    ─── [CLS] ───
      • ตำแหน่งแรกของทุก sequence
      • หลัง forward pass → hidden state ของ [CLS] = "summary ของทั้งประโยค"
      • Classification head ใช้ h_[CLS] (768-dim) → predict class
      • ดู source's BERTForClassification.predict_proba():
          h_cls = encoder.forward(...)   ← เอา [CLS] hidden state
          probs = head.forward(h_cls)    ← classify จาก vector เดียว

    ─── token_ids ───
      • vocab id ของแต่ละ token จริงในประโยค
      • ผ่าน embedding layer → vector → attention layers
      • แต่ละ token ดู context จาก token อื่น ๆ ในประโยค (self-attention)

    ─── [SEP] ───
      • บอก "จบประโยค" — ใน Next Sentence Prediction (NSP)
        แยก 2 ประโยค: [CLS] sent_A [SEP] sent_B [SEP]
      • Single sentence task ก็ยังต้องมี (BERT คาดหวัง pattern นี้)

    ─── [PAD] ───
      • เติมให้ทุก sequence ใน batch ยาว = max_length
      • GPU ต้อง process tensor ขนาดเดียวกัน → ต้อง pad
      • ❗ PAD ต้องถูก mask ใน self-attention (ดู (mm)) ไม่งั้น noise

  ▶ ตัวอย่างจาก output ด้านบน:
    'จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์'
    → [1, ...token ids 30 ตัว..., 2]   real_len = 32, ไม่ pad
    → ถ้า text สั้นกว่า 30 chars จะมี [0, 0, 0, ...] ต่อท้าย
""")

# ================================================================
# (mm) attention_mask 1 vs 0 — ต่างกันยังไง — BERT ใช้ mask ใน self-attention อย่างไร
# W 4.1 : MM attention_mask=1 vs 0 ต่างกันอย่างไร — BERT ใช้ mask นี้ตรงไหนใน self-attention
# ================================================================
print("=" * 70)
print("  (mm) attention_mask = 1 vs 0")
print("=" * 70)

print(f"""
  ▶ Pattern จาก source:
      mask = [1] * len(real_ids) + [0] * pad_len
      ids  = real_ids + [0] * pad_len

    1 = "real token"  → ให้ attention มอง
    0 = "padding"     → ห้ามให้ attention มอง

  ▶ BERT ใช้ mask ที่ scaled-dot-product attention:

      scores = Q @ K.T / sqrt(d_k)         # shape (seq, seq)
      scores = scores.masked_fill(
                 mask==0, -1e9             # หรือ -inf
               )
      weights = softmax(scores)            # PAD position → ≈ 0

    ผลคือ:
      - ทุก query token ที่ attend ไปยัง [PAD] จะได้ weight ≈ 0
      - [PAD] ก็ไม่มีส่วนร่วมในการสร้าง output ของ token จริง

  ▶ ทำไมต้อง mask:

    1. Padding ไม่มี semantic meaning (เป็น 0 ทั้ง vector)
       ถ้าไม่ mask → softmax จะแบ่ง weight ให้ PAD ด้วย → noise

    2. Inconsistent batch length:
       text สั้นใน batch จะมี PAD เยอะ → ถ้าไม่ mask weight จะถูก
       "เจือจาง" ตามจำนวน PAD → ผล classification ไม่ stable

    3. ตอน fine-tune: ถ้า PAD เข้า loss → gradient เปื้อน

  ▶ ตัวอย่างจาก output:
    'บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้อง'  (33 chars → ตัดเหลือ 30)
    real_tokens = 32 → mask = [1, 1, 1, ..., 1] (ไม่มี PAD เพราะเต็ม max_length)

    ถ้าใช้ max_length=64:
    real = 32, pad = 32 → mask = [1×32, 0×32]
    → BERT จะ attend แค่ 32 ตัวแรก ละทิ้ง 32 ตัวหลัง
""")

# ================================================================
# (nn) ทำไม WangchanBERTa เหมาะกว่า mBERT สำหรับ dataset นี้
# W 4.1 : NN ทำไม WangchanBERTa ถึงเหมาะกว่า mBERT สำหรับ dataset นี้ — เปรียบเทียบ Thai coverage 1% vs 100%
# ================================================================
print("=" * 70)
print("  (nn) WangchanBERTa vs mBERT สำหรับ Thai legal dataset")
print("=" * 70)

print(f"\n  ▶ MODEL_REGISTRY (จาก source):")
print(f"  {'Model':<16} {'Params':<8} {'Thai%':<8} หมายเหตุ")
print(f"  {'─'*56}")
for name, m in MODEL_REGISTRY.items():
    print(f"  {name:<16} {m['params']:<8} {m['thai_cov']:<8} {m['note']}")

print(f"""
  ▶ Thai coverage 1% (mBERT) vs 100% (WangchanBERTa):

    mBERT (~1% Thai):
      • Pre-train บน Wikipedia 104 ภาษา → Thai เป็น minority
      • Vocabulary มีคำไทยน้อย → คำไทยส่วนใหญ่ถูกตัดเป็น sub-word
        เช่น 'สิทธิบัตร' → ['สิ', '##ทธิ', '##บั', '##ตร']
      • Embedding ของ sub-word ไทยเรียนรู้จากข้อมูลน้อย → คุณภาพต่ำ

    WangchanBERTa (100% Thai):
      • Pre-train บน Thai Wikipedia + CCNet (corpus ภาษาไทยล้วน)
      • Vocabulary มีคำไทยครบ → 'สิทธิบัตร' = 1 token
      • Embedding คำไทยมาจากข้อมูล Thai หลายล้านประโยค

  ▶ ผลในงานของเรา (Thai IP Legal Classification):

    1. Tokenization quality:
       mBERT:           'สิทธิบัตรการประดิษฐ์' → 8-12 sub-words ❌
       WangchanBERTa:   'สิทธิบัตรการประดิษฐ์' → 2-3 tokens ✅
       → max_length เดียวกัน, WangchanBERTa เก็บข้อความได้มากกว่า

    2. Semantic representation:
       mBERT: sub-word ของศัพท์กฎหมายไทย → embedding ไม่ specialized
       WangchanBERTa: คำกฎหมายเรียนรู้จาก Thai Wikipedia/news แล้ว
                       → embedding ใกล้เคียง semantic จริง

    3. Fine-tune efficiency:
       mBERT ต้องเรียนรู้ทั้ง "ภาษาไทย" + "domain กฎหมาย" พร้อมกัน
       → ใช้ data + epoch มากกว่ากว่าจะ converge
       WangchanBERTa รู้ภาษาไทยอยู่แล้ว → fine-tune แค่ "domain knowledge"
       → ใช้ data 1/5 ก็ได้ผลเทียบเท่า

    4. Same params (110M) แต่คุณภาพต่างกันชัด:
       mBERT 110M → 1.1M params 'จริง' ที่เกี่ยวกับ Thai (1%)
       WangchanBERTa 110M → 110M params ทุ่มกับ Thai (100%)
       → "size เท่ากัน, capacity ต่างกัน 100 เท่า"

  ▶ ข้อสรุป:
    สำหรับ dataset ภาษาไทยล้วน (เช่น Thai IP Legal Corpus 66 ตัวอย่าง)
    WangchanBERTa เป็น strong baseline ก่อนคิดถึง model ใหญ่กว่า
    mBERT เหมาะกับ multilingual task (เช่น แปลไทย-อังกฤษ-จีน)

    → source's note ที่ MODEL_REGISTRY['wangchanberta']:
      "ดีที่สุดสำหรับ Thai legal text" ✓ (sound choice)
""")


"""

======================================================================
  MockTokenize.show() บน 3 ประโยคจาก dataset
======================================================================

  ข้อความ    : 'จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์'
  input_ids  : [1, 3692, 3735, 3748, 3721, 3718, 3712, 3721, 3736, 3705, 3726, 3736, 3709, 3688, 3757, 3734, 3707, 3737, 3756, 3748, 3721, 3737, 3718, 3709, 3749, 3710, 3710, 3726, 3736, 3707, 3708, 2]
  mask       : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  real tokens: 32  PAD: 0

  แยกส่วน:
    [CLS] = 1
    tokens= [3692, 3735, 3748, 3721, 3718, 3712, 3721, 3736, 3705, 3726, 3736, 3709, 3688, 3757, 3734, 3707, 3737, 3756, 3748, 3721, 3737, 3718, 3709, 3749, 3710, 3710, 3726, 3736, 3707, 3708]
    [SEP] = 2
    [PAD] = []

  ข้อความ    : 'บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้อง'
  input_ids  : [1, 3710, 3719, 3736, 3725, 3733, 3707, 3752, 3704, 3757, 3719, 3733, 3710, 3729, 3709, 3740, 3697, 3734, 3705, 3751, 3727, 3757, 3751, 3694, 3757, 3726, 3736, 3707, 3708, 3736, 3710, 2]
  mask       : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  real tokens: 32  PAD: 0

  แยกส่วน:
    [CLS] = 1
    tokens= [3710, 3719, 3736, 3725, 3733, 3707, 3752, 3704, 3757, 3719, 3733, 3710, 3729, 3709, 3740, 3697, 3734, 3705, 3751, 3727, 3757, 3751, 3694, 3757, 3726, 3736, 3707, 3708, 3736, 3710]
    [SEP] = 2
    [PAD] = []

  ข้อความ    : 'ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์'
  input_ids  : [1, 3712, 3741, 3757, 3705, 3757, 3729, 3691, 3727, 3734, 3707, 3735, 3695, 3757, 3735, 3750, 3711, 3719, 3749, 3685, 3719, 3717, 3688, 3729, 3717, 3714, 3736, 3723, 3748, 3705, 3729, 2]
  mask       : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  real tokens: 32  PAD: 0

  แยกส่วน:
    [CLS] = 1
    tokens= [3712, 3741, 3757, 3705, 3757, 3729, 3691, 3727, 3734, 3707, 3735, 3695, 3757, 3735, 3750, 3711, 3719, 3749, 3685, 3719, 3717, 3688, 3729, 3717, 3714, 3736, 3723, 3748, 3705, 3729]
    [SEP] = 2
    [PAD] = []

======================================================================
  (ll) input_ids structure — บทบาทใน BERT
======================================================================

  ▶ Pattern: [CLS] + token_ids + [SEP] + [PAD]...

    [CLS] = 1   "Classification token" — token พิเศษที่อยู่ตำแหน่งแรกเสมอ
    [SEP] = 2   "Separator" — ปิดท้ายประโยค (หรือคั่นระหว่าง 2 ประโยค)
    [PAD] = 0   "Padding" — เติมให้ทุก sequence ยาวเท่ากัน
    token = {อื่น ๆ}  vocab id ของแต่ละ token

  ▶ บทบาทใน BERT:

    ─── [CLS] ───
      • ตำแหน่งแรกของทุก sequence
      • หลัง forward pass → hidden state ของ [CLS] = "summary ของทั้งประโยค"
      • Classification head ใช้ h_[CLS] (768-dim) → predict class
      • ดู source's BERTForClassification.predict_proba():
          h_cls = encoder.forward(...)   ← เอา [CLS] hidden state
          probs = head.forward(h_cls)    ← classify จาก vector เดียว

    ─── token_ids ───
      • vocab id ของแต่ละ token จริงในประโยค
      • ผ่าน embedding layer → vector → attention layers
      • แต่ละ token ดู context จาก token อื่น ๆ ในประโยค (self-attention)

    ─── [SEP] ───
      • บอก "จบประโยค" — ใน Next Sentence Prediction (NSP)
        แยก 2 ประโยค: [CLS] sent_A [SEP] sent_B [SEP]
      • Single sentence task ก็ยังต้องมี (BERT คาดหวัง pattern นี้)

    ─── [PAD] ───
      • เติมให้ทุก sequence ใน batch ยาว = max_length
      • GPU ต้อง process tensor ขนาดเดียวกัน → ต้อง pad
      • ❗ PAD ต้องถูก mask ใน self-attention (ดู (mm)) ไม่งั้น noise

  ▶ ตัวอย่างจาก output ด้านบน:
    'จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์'
    → [1, ...token ids 30 ตัว..., 2]   real_len = 32, ไม่ pad
    → ถ้า text สั้นกว่า 30 chars จะมี [0, 0, 0, ...] ต่อท้าย

======================================================================
  (mm) attention_mask = 1 vs 0
======================================================================

  ▶ Pattern จาก source:
      mask = [1] * len(real_ids) + [0] * pad_len
      ids  = real_ids + [0] * pad_len

    1 = "real token"  → ให้ attention มอง
    0 = "padding"     → ห้ามให้ attention มอง

  ▶ BERT ใช้ mask ที่ scaled-dot-product attention:

      scores = Q @ K.T / sqrt(d_k)         # shape (seq, seq)
      scores = scores.masked_fill(
                 mask==0, -1e9             # หรือ -inf
               )
      weights = softmax(scores)            # PAD position → ≈ 0

    ผลคือ:
      - ทุก query token ที่ attend ไปยัง [PAD] จะได้ weight ≈ 0
      - [PAD] ก็ไม่มีส่วนร่วมในการสร้าง output ของ token จริง

  ▶ ทำไมต้อง mask:

    1. Padding ไม่มี semantic meaning (เป็น 0 ทั้ง vector)
       ถ้าไม่ mask → softmax จะแบ่ง weight ให้ PAD ด้วย → noise

    2. Inconsistent batch length:
       text สั้นใน batch จะมี PAD เยอะ → ถ้าไม่ mask weight จะถูก
       "เจือจาง" ตามจำนวน PAD → ผล classification ไม่ stable

    3. ตอน fine-tune: ถ้า PAD เข้า loss → gradient เปื้อน

  ▶ ตัวอย่างจาก output:
    'บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้อง'  (33 chars → ตัดเหลือ 30)
    real_tokens = 32 → mask = [1, 1, 1, ..., 1] (ไม่มี PAD เพราะเต็ม max_length)

    ถ้าใช้ max_length=64:
    real = 32, pad = 32 → mask = [1×32, 0×32]
    → BERT จะ attend แค่ 32 ตัวแรก ละทิ้ง 32 ตัวหลัง

======================================================================
  (nn) WangchanBERTa vs mBERT สำหรับ Thai legal dataset
======================================================================

  ▶ MODEL_REGISTRY (จาก source):
  Model            Params   Thai%    หมายเหตุ
  ────────────────────────────────────────────────────────
  mbert            110M     ~1%      ใช้ได้หลายภาษา แต่ Thai coverage น้อย
  xlmr             270M     ~5%      RoBERTa architecture, Thai ดีกว่า mBERT
  wangchanberta    110M     100%     ดีที่สุดสำหรับ Thai legal text

  ▶ Thai coverage 1% (mBERT) vs 100% (WangchanBERTa):

    mBERT (~1% Thai):
      • Pre-train บน Wikipedia 104 ภาษา → Thai เป็น minority
      • Vocabulary มีคำไทยน้อย → คำไทยส่วนใหญ่ถูกตัดเป็น sub-word
        เช่น 'สิทธิบัตร' → ['สิ', '##ทธิ', '##บั', '##ตร']
      • Embedding ของ sub-word ไทยเรียนรู้จากข้อมูลน้อย → คุณภาพต่ำ

    WangchanBERTa (100% Thai):
      • Pre-train บน Thai Wikipedia + CCNet (corpus ภาษาไทยล้วน)
      • Vocabulary มีคำไทยครบ → 'สิทธิบัตร' = 1 token
      • Embedding คำไทยมาจากข้อมูล Thai หลายล้านประโยค

  ▶ ผลในงานของเรา (Thai IP Legal Classification):

    1. Tokenization quality:
       mBERT:           'สิทธิบัตรการประดิษฐ์' → 8-12 sub-words ❌
       WangchanBERTa:   'สิทธิบัตรการประดิษฐ์' → 2-3 tokens ✅
       → max_length เดียวกัน, WangchanBERTa เก็บข้อความได้มากกว่า

    2. Semantic representation:
       mBERT: sub-word ของศัพท์กฎหมายไทย → embedding ไม่ specialized
       WangchanBERTa: คำกฎหมายเรียนรู้จาก Thai Wikipedia/news แล้ว
                       → embedding ใกล้เคียง semantic จริง

    3. Fine-tune efficiency:
       mBERT ต้องเรียนรู้ทั้ง "ภาษาไทย" + "domain กฎหมาย" พร้อมกัน
       → ใช้ data + epoch มากกว่ากว่าจะ converge
       WangchanBERTa รู้ภาษาไทยอยู่แล้ว → fine-tune แค่ "domain knowledge"
       → ใช้ data 1/5 ก็ได้ผลเทียบเท่า

    4. Same params (110M) แต่คุณภาพต่างกันชัด:
       mBERT 110M → 1.1M params 'จริง' ที่เกี่ยวกับ Thai (1%)
       WangchanBERTa 110M → 110M params ทุ่มกับ Thai (100%)
       → "size เท่ากัน, capacity ต่างกัน 100 เท่า"

  ▶ ข้อสรุป:
    สำหรับ dataset ภาษาไทยล้วน (เช่น Thai IP Legal Corpus 66 ตัวอย่าง)
    WangchanBERTa เป็น strong baseline ก่อนคิดถึง model ใหญ่กว่า
    mBERT เหมาะกับ multilingual task (เช่น แปลไทย-อังกฤษ-จีน)

    → source's note ที่ MODEL_REGISTRY['wangchanberta']:
      "ดีที่สุดสำหรับ Thai legal text" ✓ (sound choice)

"""
