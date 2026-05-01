import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import json
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

# ===== Import จาก w4 source — suppress module-level prints =====
with redirect_stdout(io.StringIO()):
    from w4_bert_finetune import (
        BERTForClassification,
        MockBERTEncoder,
        ClassificationHead,
    )

# ===== Load THAI_IP_DATASET จาก JSON (per architecture pipeline) =====
DATA_PATH = Path(__file__).parent / "data" / "processed" / "thai_ip_dataset.json"
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
THAI_IP_DATASET = data["samples"]   # list ของ {"id", "text", "label", "label_name"}

# ===== โค้ดบังคับตามโจทย์ =====
print("=" * 70)
print("  รัน BERTForClassification บน dataset ทั้งหมด (66 ตัวอย่าง)")
print("=" * 70)
model = BERTForClassification(seed=42)
all_texts = [d["text"] for d in THAI_IP_DATASET]
model.show_prediction(all_texts)

print("\n" + "=" * 70)
print("  วิเคราะห์ confidence distribution")
print("=" * 70)
enc = model.tokenizer.encode(all_texts, max_length=32)
prob = model.predict_proba(enc["input_ids"], enc["attention_mask"])
conf = prob.max(axis=1)
print(f"Mean confidence: {conf.mean():.3f}")
print(f"Low confidence (<0.5): {(conf < 0.5).sum()} cases")

# ================================================================
# (ss) อธิบาย architecture: MockBERTEncoder → h_cls (768 dim) → ClassificationHead → softmax
# W 4.3 : SS อธิบาย architecture: MockBERTEncoder → h_cls (768 dim) → ClassificationHead → softmax — แต่ละส่วนทำอะไร
# ================================================================
print("\n" + "=" * 70)
print("  (ss) Architecture: MockBERTEncoder → h_cls → ClassificationHead → softmax")
print("=" * 70)

print(f"""
  ▶ Pipeline diagram:

    text → tokenizer → input_ids (batch, 32)
                       attention_mask (batch, 32)
                            │
                            ▼
                    ┌──────────────────┐
                    │ MockBERTEncoder  │  ← 12-layer Transformer (mock)
                    │ (768 hidden_size)│     ใน production = bert-base
                    └────────┬─────────┘
                             │
                             ▼
                    h_cls (batch, 768)    ← representation ของ [CLS]
                             │
                             ▼
                    ┌──────────────────┐
                    │ClassificationHead│  ← Linear(768 → 3) + Dropout
                    │   W (3, 768)     │
                    │   b (3,)         │
                    └────────┬─────────┘
                             │
                             ▼
                    logits (batch, 3)     ← raw scores
                             │
                             ▼
                       softmax(logits)
                             │
                             ▼
                    probs (batch, 3)      ← sum=1 ต่อ row
                             │
                             ▼
                       argmax → predicted class

  ▶ บทบาทของแต่ละส่วน:

    ─── 1. MockBERTEncoder ───
      input  : input_ids (batch, seq_len), attention_mask (batch, seq_len)
      output : h_cls (batch, 768)  ← เก็บเฉพาะ [CLS] token เท่านั้น

      ใน production: stack ของ 12 Transformer encoder layers
        - Multi-head self-attention (W3)
        - Pre-Norm + residual
        - Feed-forward network
      ใน MOCK: คืน random vector — ไม่ได้ encode จริง (stub สำหรับ pipeline test)

      ดู source's MockBERTEncoder.forward():
        h_cls = self.rng.randn(batch, 768) * 0.1
        ← ทุก call สุ่มใหม่ (state ของ rng เปลี่ยน)

    ─── 2. h_cls (768 dim) ───
      เป็น "summary vector" ของทั้งประโยค
      ทำไม [CLS] = summary: ดูข้อ (tt)
      ใช้ 768 dim เพราะ bert-base default; bert-large ใช้ 1024

    ─── 3. ClassificationHead ───
      Linear projection: 768 → 3 (n_classes)
      logits = h_cls @ W.T + b
        W shape = (3, 768)
        b shape = (3,)
      Dropout (0.1): ทำงานเฉพาะ training=True (ดูข้อ uu)

      Source code:
        s = sqrt(2.0 / (768 + 3))  ← He initialization
        W = randn(3, 768) * s
        b = zeros(3)

    ─── 4. Softmax ───
      แปลง logits (real numbers) → probabilities (sum=1)
      formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)
      Source ใช้ "subtract max" trick เพื่อ numerical stability:
        e = exp(logits - logits.max(axis=-1, keepdims=True))
        prob = e / e.sum(axis=-1, keepdims=True)
""")

# ================================================================
# (tt) ทำไม BERT ใช้ [CLS] เป็นตัวแทนทั้งประโยค — เชื่อมกับ mean pooling ใน MiniBERT (W3)
# W 4.3 : TT ทำไม BERT ใช้ [CLS] token เป็นตัวแทนของทั้งประโยค — เชื่อมกับ mean pooling ใน MiniBERT ของ W3
# ================================================================
print("=" * 70)
print("  (tt) [CLS] vs mean pooling")
print("=" * 70)

print(f"""
  ▶ ทำไม BERT ใช้ [CLS]:

    1. Self-attention ทำให้ [CLS] เห็นทุก token:
       [CLS] อยู่ตำแหน่งที่ 0 → ใน self-attention มันสามารถ attend ไปยัง
       ทุก token (รวม [SEP], [PAD] ที่ถูก mask)
       → หลัง 12 layers, h_[CLS] เป็น "weighted aggregation" ของทุก token

    2. Pre-trained as classification anchor:
       BERT pre-train ด้วย NSP (Next Sentence Prediction)
       ใช้ [CLS] → linear → 2-class (is next sentence?)
       → [CLS] embedding เรียนรู้แล้วว่าต้อง "encode global context"
       → fine-tune downstream task ได้ทันทีโดย swap classification head

    3. ไม่ต้อง pool/aggregate เพิ่ม:
       Output ของ encoder = (batch, seq_len, hidden)
       ใช้ [CLS] = แค่ slice [:, 0, :] → ได้ (batch, hidden) ทันที
       Simple & efficient

  ▶ เปรียบเทียบกับ MiniBERT ใน W3 (mean pooling):

    Source W3 — MiniBERT.forward (line 217):
      ctx = X.mean(axis=0)              ← เฉลี่ยทุก token
      probs = softmax(W_out @ ctx + b)

    Source W4 — BERTForClassification:
      h_cls = encoder(input_ids, mask)  ← เอาแค่ [CLS]
      probs = softmax(h_cls @ W.T + b)

    เปรียบเทียบ:

    ┌───────────────────┬──────────────────┬──────────────────┐
    │ Aspect             │ MiniBERT (W3)     │ BERT (W4)         │
    ├───────────────────┼──────────────────┼──────────────────┤
    │ Pooling             │ mean(axis=0)      │ [CLS] token       │
    │ Use of special tok  │ ไม่มี [CLS]        │ มี [CLS]           │
    │ Pre-training        │ ไม่ได้ pretrain    │ pretrain แล้ว      │
    │ Equal weight tokens│ ✓ ทุก token เท่ากัน │ ✗ [CLS] aggregate  │
    │                     │                   │   ผ่าน attention    │
    │ Sensitivity to PAD  │ สูง (ถ้าไม่ mask)  │ ต่ำ (mask อยู่ใน atten) │
    └───────────────────┴──────────────────┴──────────────────┘

  ▶ ทำไม [CLS] ดีกว่า mean pooling (ในเชิง learned):
    Mean pooling: ทุก token น้ำหนักเท่ากัน (รวม PAD, stopwords)
    [CLS]: เรียนรู้ว่า "ควรให้น้ำหนักไหนกับ token ไหน" ตอน pre-train

  ▶ แต่บางงาน mean pooling ดีกว่า:
    Sentence-BERT (สำหรับ similarity) ใช้ mean pooling
    เพราะ [CLS] ไม่ได้ optimize สำหรับ similarity task
    → ตอบคำถาม: ใช้ [CLS] หรือ mean ขึ้นอยู่กับ downstream task
""")

# ================================================================
# (uu) Dropout ทำงานอย่างไร — training=True vs training=False ต่างกันอย่างไร
# W 4.3 : UU Dropout ใน ClassificationHead ทำงานอย่างไร — training=True กับ training=False ต่างกันอย่างไร
# ================================================================
print("=" * 70)
print("  (uu) Dropout: training=True vs training=False")
print("=" * 70)

print(f"""
  ▶ Source code (ClassificationHead.forward):

    def forward(self, h_cls, training=False):
        if training:
            mask = (np.random.rand(*h_cls.shape) > self.dropout).astype(np.float32)
            h_cls = h_cls * mask / (1 - self.dropout)
        logits = h_cls @ self.W.T + self.b
        ...

  ▶ พฤติกรรม:

    ─── training=True (ตอน train) ───
      1. สุ่ม binary mask: 1 ด้วย probability (1-dropout), 0 ด้วย dropout
      2. h_cls *= mask  → ค่า ~10% ถูกเซ็ตเป็น 0 (drop)
      3. หาร (1 - dropout) → "scale up" เพื่อชดเชยค่าที่หายไป
         (เรียก inverted dropout — keep expected value คงที่)

    ─── training=False (ตอน inference) ───
      1. ไม่ทำอะไร — ใช้ h_cls เต็ม ๆ
      2. ไม่มี randomness → ผลลัพธ์ deterministic
""")

# Demo: แสดงผลของ dropout
print(f"  ▶ Demo เปรียบเทียบ:")
np.random.seed(0)
demo_h = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
head = ClassificationHead(hidden_size=5, n_classes=3, dropout=0.4, seed=42)

prob_eval = head.forward(demo_h, training=False)
print(f"\n  h_cls            = {demo_h[0]}")
print(f"  training=False:")
print(f"    h_cls (unchanged) = {demo_h[0]}")
print(f"    prob               = {prob_eval[0].round(3)}")

print(f"\n  training=True (3 ครั้ง — ดูความแตกต่าง):")
for trial in range(3):
    np.random.seed(trial * 7)
    mask = (np.random.rand(*demo_h.shape) > 0.4).astype(np.float32)
    h_dropped = demo_h * mask / (1 - 0.4)
    print(f"    Trial {trial+1}: h_cls (dropped) = {h_dropped[0].round(2)}")

print(f"""
  ▶ ทำไมต้องใช้ Dropout:

    1. ป้องกัน Overfitting:
       Network จะไม่ "พึ่งพา" feature ใด feature หนึ่งมากเกินไป
       เพราะ feature นั้นอาจถูก drop ใน next batch

    2. Implicit ensemble:
       แต่ละ training step = train sub-network ที่ random
       → final model = ensemble ของ sub-networks
       → robust กว่า single network

    3. ทำไมต้อง scale ด้วย 1/(1-dropout):
       ตอน train: expected value = h_cls * (1 - dropout)
       ตอน eval:  ใช้ h_cls เต็ม (1.0×)
       → mismatch! แก้โดย scale up ตอน train ให้ expected value คงที่
       → ตอน eval ไม่ต้องทำอะไรเพิ่ม

  ▶ ใน source ใช้ dropout=0.1 (10%)
    เป็นค่ามาตรฐานของ BERT — สูงไป (0.5) จะ underfit, ต่ำไป (0.0) overfit
""")

# ================================================================
# (vv) เปรียบเทียบ W2 (BiLSTM) vs W4 (BERT) — BERT ได้เปรียบยังไง MockBERTEncoder ขาดอะไร
# W 4.3 : VV เปรียบเทียบ performance W2 (BiLSTM) กับ W4 (BERT) — BERT ได้เปรียบในด้านใด และ MockBERTEncoder ขาดอะไรจาก BERT จริง
# ================================================================
print("=" * 70)
print("  (vv) W2 BiLSTM vs W4 BERT — และ MockBERTEncoder ขาดอะไร")
print("=" * 70)

print(f"""
  ▶ เปรียบเทียบ architecture:

    ┌────────────────────┬──────────────────────┬─────────────────────┐
    │ Aspect              │ W2 BiLSTM             │ W4 BERT              │
    ├────────────────────┼──────────────────────┼─────────────────────┤
    │ Sequence model      │ Recurrent (เรียงลำดับ) │ Self-attention       │
    │ Long-range dep.     │ ปานกลาง (vanishing)    │ ดี (direct attention) │
    │ Parallelization     │ ❌ sequential          │ ✅ all positions     │
    │ Pre-training        │ ❌ train from scratch  │ ✅ pretrain ใน corpus│
    │                     │                       │   ขนาดใหญ่           │
    │ Bidirectional       │ ✅ (forward+backward)  │ ✅ (full attention)  │
    │ Param count         │ ~50K (เล็ก)            │ 110M (ใหญ่)          │
    │ Train time           │ เร็ว                   │ ช้า                  │
    │ Inference            │ เร็ว                   │ ช้า                  │
    │ Domain knowledge     │ เริ่มจาก 0             │ มี world knowledge    │
    └────────────────────┴──────────────────────┴─────────────────────┘

  ▶ BERT ได้เปรียบ BiLSTM ในด้าน:

    1. Pre-trained representation:
       BERT รู้ภาษาแล้ว (syntax, semantics, world knowledge)
       BiLSTM ต้องเรียนทุกอย่างจาก dataset 66 ตัวอย่าง → underfit

    2. Long-range dependency:
       BiLSTM 'จำ' ผ่าน hidden state ที่ค่อย ๆ จาง
       BERT attention เชื่อมตรงระหว่างคู่ใด ๆ ในประโยค → ไม่จาง

    3. Contextualized embedding:
       BiLSTM: embedding ของ 'ละเมิด' = vector เดียวเสมอ
       BERT: embedding ของ 'ละเมิด' เปลี่ยนตาม context รอบข้าง
             ('ละเมิดสิทธิบัตร' vs 'ไม่ละเมิด')

    4. Transfer learning:
       Fine-tune BERT บน 66 ตัวอย่าง → ใช้ได้
       Train BiLSTM from scratch บน 66 ตัวอย่าง → overfit

  ▶ MockBERTEncoder ขาดอะไรจาก BERT จริง:

    Source: h_cls = self.rng.randn(batch, 768) * 0.1
            ← random! ไม่ใช่ encoding จริง

    ขาดสิ่งเหล่านี้:

    1. Token Embeddings (30K vocab × 768 dim)
       BERT จริง: lookup vocab → embedding ที่เรียนรู้
       Mock: ไม่มี vocab จริง

    2. Position Embeddings (max 512 × 768)
       BERT จริง: เพิ่ม positional info
       Mock: ไม่สนใจ position

    3. Segment Embeddings (2 × 768)
       BERT จริง: บอก "อยู่ประโยคแรกหรือสอง"
       Mock: ไม่รองรับ

    4. 12 Transformer Layers
       BERT จริง: 12 × (Multi-head attention + FFN + LN)
       Mock: ข้ามทั้งหมด

    5. Pre-training knowledge
       BERT จริง: train บน 16GB ของ Wikipedia + BookCorpus
       Mock: random number generator

    6. Fine-tuning capability
       BERT จริง: gradient flows → weights update
       Mock: ไม่มี gradient → no learning

  ▶ ทำไมยังใช้ MockBERTEncoder:
    - ทดสอบ pipeline / API design ก่อน install transformers
    - ใน CI/CD / unit test (เร็วและไม่ต้องโหลด model)
    - สอน concept โดยไม่ต้องรอ download model จริง

    ⚠️  ผลที่ได้จาก mock จึงไม่ meaningful — confidence ที่เห็นคือ
        random distribution ของ softmax(random vector @ random matrix)
""")

# ================================================================
# (ww) ถ้าจะนำโมเดล W4 ไปใช้จริงในศาลทรัพย์สินทางปัญญา ต้องทำอะไรอย่างน้อย 3 ขั้นตอน
# W 4.3 : WW ถ้าจะนำโมเดล W4 ไปใช้จริงในศาลทรัพย์สินทางปัญญา ต้องทำอะไรเพิ่มเติมอย่างน้อย 3 ขั้นตอน
# ================================================================
print("=" * 70)
print("  (ww) Production Roadmap — นำ W4 ไปใช้จริงในศาล")
print("=" * 70)

print(f"""
  ▶ 6 ขั้นตอน (เกินขั้นต่ำ 3 ตามโจทย์):

    ─── ขั้น 1: เปลี่ยน MockBERTEncoder → WangchanBERTa ───────────────
      • โหลด airesearch/wangchanberta-base-att-spm-uncased
      • ติดตั้ง transformers, torch
      • ทดสอบ tokenizer + inference บน sample จริง
      • Benchmark latency (target < 200ms/case)
      เหตุผล: ตามที่ W4.1 (nn) อธิบาย — Thai coverage 100% > 1%

    ─── ขั้น 2: รวบรวม Gold Standard Dataset (≥ 1,000 cases) ────────
      • ปัจจุบัน 66 ตัวอย่าง = "Bronze" (ไม่พอ train)
      • ขอความร่วมมือศาลทรัพย์สินทางปัญญา → ขอคำพิพากษาจริง
      • Annotator คือนักกฎหมาย IP (ไม่ใช่นักศึกษา) — Inter-annotator
        agreement ต้อง κ ≥ 0.8
      • Split: 70% train / 15% val / 15% test
      • Stratify by class + by court level (ศาลชั้นต้น/อุทธรณ์/ฎีกา)

    ─── ขั้น 3: Warm-up + Fine-tune (ตาม W4.2 pp) ──────────────────
      • Phase 1: Freeze BERT, train embedding ของ legal terms ใหม่ 3 epochs
      • Phase 2: Unfreeze ทุก layer, LR=2e-5, 5-10 epochs
      • Phase 3: Early stopping ตาม val loss (patience=3)
      • Cross-validation 5-fold → estimate generalization

    ─── ขั้น 4: Calibration + Confidence Threshold ────────────────
      • Temperature scaling: ปรับ softmax ให้ confidence ตรงความเป็นจริง
      • Set threshold: cases ที่ confidence < 0.85 → ให้นักกฎหมาย review
        (ไม่ auto-decide)
      • Add abstention class: model ตอบ "ไม่แน่ใจ" ได้

    ─── ขั้น 5: Explainability (XAI) — Required by Law ─────────────
      • Attention visualization: แสดง token ที่ model พิจารณา
      • SHAP / LIME: counterfactual explanation
        ("ถ้าไม่มีคำว่า 'ละเมิด' → prediction เปลี่ยนเป็น...")
      • Cite legal precedents ที่ใกล้เคียง (retrieval-augmented)
      • รายงาน confidence interval ไม่ใช่จุดเดียว
      เหตุผล: กฎหมายต้องการ "อธิบายได้" — black-box decision = invalid

    ─── ขั้น 6: Audit + Bias Testing ─────────────────────────────
      • Test fairness across:
         - บริษัทใหญ่ vs บริษัทเล็ก (defendant size)
         - บริษัทไทย vs ต่างชาติ (nationality bias)
         - คดีเก่า vs คดีใหม่ (temporal bias)
      • Adversarial robustness: ทดสอบ paraphrase / typo
      • Audit logs: log ทุก prediction + reasoning เพื่อ accountability
      • Periodic retraining: ทุก 6 เดือนด้วย data ใหม่

  ▶ Deployment architecture:

      Lawyer/Clerk Input
            │
            ▼
      [Pre-screening BERT]  ← W4 model
            │
       confidence ≥ 0.85?
       ┌────┴────┐
       │         │
       ▼         ▼
    Auto      Human Review
    Categorize  by IP Lawyer
       │         │
       └────┬────┘
            ▼
       Audit Log
       Final Decision

  ▶ ⚠️  สิ่งที่ห้ามทำ:
    ❌ ใช้ model auto-tัdsin คดี (decision making)
    ❌ ปกปิด: เปิดเผยว่าใช้ AI ทุกครั้ง
    ❌ ลืม update: data drift ทำให้ accuracy ตก
    ❌ ใช้ MockBERTEncoder ใน production!!! (mock = random)
""")


"""

======================================================================
  รัน BERTForClassification บน dataset ทั้งหมด (66 ตัวอย่าง)
======================================================================

  ข้อความ                                       การทำนาย             ความมั่นใจ
  ───────────────────────────────────────────────────────────────────────────
  จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิ   ละเมิด_สิทธิบัตร     95.9%
  ผู้ต้องหานำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจา   ละเมิด_ลิขสิทธิ์     96.0%
  บริษัทจำเลยผลิตยาสามัญโดยละเมิดสิทธิบัตรยาต   ไม่ละเมิด            95.7%
  จำเลยนำเทคโนโลยีจดสิทธิบัตรไปใช้เชิงพาณิชย์   ละเมิด_สิทธิบัตร     38.6%
  ผู้ต้องหาผลิตอุปกรณ์อิเล็กทรอนิกส์เลียนแบบส   ละเมิด_สิทธิบัตร     38.6%
  จำเลยขายสินค้าปลอมแปลงที่ใช้กระบวนการผลิตตา   ละเมิด_สิทธิบัตร     36.7%
  บริษัทนำเข้าผลิตภัณฑ์ที่ละเมิดอนุสิทธิบัตรข   ละเมิด_สิทธิบัตร     35.0%
  จำเลยผลิตเครื่องจักรโดยใช้กลไกที่ได้รับสิทธ   ไม่ละเมิด            36.1%
  ผู้ต้องหาส่งออกสินค้าที่ละเมิดสิทธิบัตรไปยั   ละเมิด_ลิขสิทธิ์     38.6%
  บริษัทจำเลยใช้สูตรเคมีที่ได้รับสิทธิบัตรในก   ไม่ละเมิด            38.2%
  จำเลยผลิตอุปกรณ์การแพทย์โดยละเมิดสิทธิบัตรโ   ละเมิด_ลิขสิทธิ์     39.4%
  ผู้ต้องหาทำซ้ำกระบวนการผลิตที่จดสิทธิบัตรแล   ละเมิด_ลิขสิทธิ์     35.3%
  บริษัทจำเลยผลิตแบตเตอรี่โดยใช้เทคโนโลยีที่ไ   ละเมิด_ลิขสิทธิ์     39.5%
  จำเลยใช้วิธีการทางวิศวกรรมที่ได้รับสิทธิบัต   ละเมิด_ลิขสิทธิ์     35.8%
  ผู้ต้องหาผลิตชิ้นส่วนยานยนต์โดยละเมิดสิทธิบ   ไม่ละเมิด            35.6%
  บริษัทจำเลยนำกระบวนการผลิตที่จดสิทธิบัตรมาใ   ละเมิด_ลิขสิทธิ์     36.5%
  จำเลยผลิตโดรนโดยใช้เทคโนโลยีที่ได้รับการจดส   ละเมิด_สิทธิบัตร     35.5%
  ผู้ต้องหาเลียนแบบการออกแบบผลิตภัณฑ์ที่ได้รั   ละเมิด_สิทธิบัตร     39.5%
  บริษัทนำเข้าเครื่องพิมพ์ 3D ที่ใช้เทคโนโลยี   ไม่ละเมิด            37.5%
  จำเลยผลิตยาปฏิชีวนะโดยใช้สูตรที่อยู่ภายใต้ส   ไม่ละเมิด            34.6%
  ผู้ต้องหาทำซ้ำกระบวนการหมักที่ได้รับสิทธิบั   ละเมิด_สิทธิบัตร     35.6%
  บริษัทจำเลยผลิตเซมิคอนดักเตอร์โดยละเมิดสิทธ   ละเมิด_สิทธิบัตร     38.8%
  จำเลยนำเข้าและจำหน่ายชิปที่ใช้สถาปัตยกรรมตา   ไม่ละเมิด            36.5%
  ผู้ต้องหาผลิตอุปกรณ์โทรคมนาคมโดยละเมิดสิทธิ   ละเมิด_ลิขสิทธิ์     37.7%
  บริษัทจำเลยใช้กระบวนการบำบัดน้ำที่ได้รับสิท   ไม่ละเมิด            37.9%
  จำเลยผลิตแผงโซลาร์โดยใช้เทคโนโลยีที่ได้รับส   ไม่ละเมิด            37.7%
  ผู้ต้องหานำเข้าอุปกรณ์ฟอกไตที่ละเมิดสิทธิบั   ละเมิด_ลิขสิทธิ์     35.1%
  บริษัทจำเลยผลิตสีอุตสาหกรรมโดยใช้สูตรที่ได้   ไม่ละเมิด            36.7%
  จำเลยใช้อัลกอริทึมที่จดสิทธิบัตรในซอฟต์แวร์   ไม่ละเมิด            34.3%
  ผู้ต้องหาผลิตอุปกรณ์ IoT โดยละเมิดสิทธิบัตร   ละเมิด_ลิขสิทธิ์     37.0%
  บริษัทนำเข้าและจำหน่ายผลิตภัณฑ์ที่ใช้วัสดุน   ละเมิด_ลิขสิทธิ์     36.8%
  จำเลยผลิตชุดทดสอบโควิดที่เลียนแบบเทคโนโลยีส   ละเมิด_สิทธิบัตร     39.2%
  ผู้ต้องหาใช้กระบวนการถลุงแร่ที่ได้รับสิทธิบ   ละเมิด_สิทธิบัตร     40.2%
  บริษัทจำเลยผลิตวัคซีนโดยละเมิดสิทธิบัตรของบ   ละเมิด_ลิขสิทธิ์     35.6%
  จำเลยนำเทคโนโลยีบล็อกเชนที่จดสิทธิบัตรไปใช้   ไม่ละเมิด            35.1%
  ผู้ต้องหาผลิตหุ่นยนต์อุตสาหกรรมโดยละเมิดสิท   ละเมิด_ลิขสิทธิ์     40.1%
  บริษัทจำเลยใช้เทคโนโลยีการพิมพ์ inkjet ที่จ   ไม่ละเมิด            34.6%
  จำเลยผลิตวัสดุก่อสร้างโดยใช้สูตรซีเมนต์ที่ไ   ละเมิด_สิทธิบัตร     33.9%
  ผู้ต้องหานำเข้ายาชีววัตถุที่ละเมิดสิทธิบัตร   ละเมิด_ลิขสิทธิ์     37.1%
  บริษัทจำเลยผลิตตัวเก็บประจุโดยใช้วัสดุไดอิเ   ไม่ละเมิด            39.1%
  จำเลยทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไ   ละเมิด_สิทธิบัตร     35.3%
  ผู้ต้องหาเผยแพร่ภาพยนตร์บน YouTube โดยละเมิ   ละเมิด_สิทธิบัตร     39.2%
  จำเลยดัดแปลงงานศิลปกรรมและนำไปจำหน่ายโดยไม่   ไม่ละเมิด            38.5%
  บริษัทจำเลยผลิตซีดีเพลงเถื่อนและจำหน่ายตามต   ละเมิด_ลิขสิทธิ์     39.3%
  ผู้ต้องหาทำซ้ำหนังสือเรียนและจำหน่ายโดยไม่ไ   ละเมิด_ลิขสิทธิ์     37.4%
  จำเลยนำภาพถ่ายของผู้เสียหายไปใช้เชิงพาณิชย์   ไม่ละเมิด            36.5%
  บริษัทดาวน์โหลดซอฟต์แวร์ไม่มีใบอนุญาตและนำไ   ละเมิด_สิทธิบัตร     36.2%
  จำเลยสตรีมเพลงโดยไม่ชำระค่าลิขสิทธิ์ให้เจ้า   ละเมิด_ลิขสิทธิ์     35.0%
  ผู้ต้องหาทำซ้ำหนังสือและจัดจำหน่ายผ่านช่องท   ละเมิด_สิทธิบัตร     37.0%
  จำเลยใช้ภาพกราฟิกที่มีลิขสิทธิ์ในโฆษณาโดยไม   ไม่ละเมิด            34.0%
  บริษัทเผยแพร่ซอฟต์แวร์เกมละเมิดลิขสิทธิ์ผ่า   ละเมิด_ลิขสิทธิ์     40.1%
  ผู้ต้องหาทำซ้ำฐานข้อมูลที่มีลิขสิทธิ์เพื่อใ   ละเมิด_ลิขสิทธิ์     33.8%
  จำเลยแปลหนังสือโดยไม่ได้รับอนุญาตและจัดพิมพ   ละเมิด_สิทธิบัตร     34.6%
  บริษัทนำเนื้อหาจากเว็บไซต์ที่มีลิขสิทธิ์มาเ   ละเมิด_ลิขสิทธิ์     39.7%
  ผู้ต้องหาเผยแพร่ภาพยนตร์ผ่าน IPTV ที่ไม่มีใ   ไม่ละเมิด            38.8%
  จำเลยทำซ้ำซอฟต์แวร์ออกแบบและจำหน่ายให้บริษั   ละเมิด_สิทธิบัตร     36.8%
  บริษัทใช้เพลงพื้นหลังในสื่อโฆษณาโดยไม่ชำระค   ไม่ละเมิด            38.3%
  ผู้ต้องหาบันทึกและแจกจ่ายการแสดงสดโดยไม่ได้   ละเมิด_สิทธิบัตร     35.9%
  จำเลยขายซอฟต์แวร์ละเมิดลิขสิทธิ์ผ่านแพลตฟอร   ไม่ละเมิด            36.5%
  บริษัทจำเลยทำซ้ำแผนที่ดิจิทัลที่มีลิขสิทธิ์   ละเมิด_ลิขสิทธิ์     36.8%
  บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้   ไม่ละเมิด            40.0%
  ผู้ผลิตชำระค่าลิขสิทธิ์ครบถ้วนตามข้อตกลง      ไม่ละเมิด            38.1%
  การใช้ซอฟต์แวร์ในขอบเขตใบอนุญาตที่ได้รับมาโ   ไม่ละเมิด            35.1%
  นักวิจัยใช้สิทธิบัตรเพื่อวัตถุประสงค์ทดลองท   ละเมิด_สิทธิบัตร     40.3%
  สิทธิบัตรหมดอายุแล้ว บริษัทจึงสามารถผลิตได้   ละเมิด_ลิขสิทธิ์     36.1%
  ศิลปินสร้างงานใหม่โดยอาศัยแนวคิดทั่วไปที่ไม   ละเมิด_สิทธิบัตร     37.7%

======================================================================
  วิเคราะห์ confidence distribution
======================================================================
Mean confidence: 0.376
Low confidence (<0.5): 66 cases

======================================================================
  (ss) Architecture: MockBERTEncoder → h_cls → ClassificationHead → softmax
======================================================================

  ▶ Pipeline diagram:

    text → tokenizer → input_ids (batch, 32)
                       attention_mask (batch, 32)
                            │
                            ▼
                    ┌──────────────────┐
                    │ MockBERTEncoder  │  ← 12-layer Transformer (mock)
                    │ (768 hidden_size)│     ใน production = bert-base
                    └────────┬─────────┘
                             │
                             ▼
                    h_cls (batch, 768)    ← representation ของ [CLS]
                             │
                             ▼
                    ┌──────────────────┐
                    │ClassificationHead│  ← Linear(768 → 3) + Dropout
                    │   W (3, 768)     │
                    │   b (3,)         │
                    └────────┬─────────┘
                             │
                             ▼
                    logits (batch, 3)     ← raw scores
                             │
                             ▼
                       softmax(logits)
                             │
                             ▼
                    probs (batch, 3)      ← sum=1 ต่อ row
                             │
                             ▼
                       argmax → predicted class

  ▶ บทบาทของแต่ละส่วน:

    ─── 1. MockBERTEncoder ───
      input  : input_ids (batch, seq_len), attention_mask (batch, seq_len)
      output : h_cls (batch, 768)  ← เก็บเฉพาะ [CLS] token เท่านั้น

      ใน production: stack ของ 12 Transformer encoder layers
        - Multi-head self-attention (W3)
        - Pre-Norm + residual
        - Feed-forward network
      ใน MOCK: คืน random vector — ไม่ได้ encode จริง (stub สำหรับ pipeline test)

      ดู source's MockBERTEncoder.forward():
        h_cls = self.rng.randn(batch, 768) * 0.1
        ← ทุก call สุ่มใหม่ (state ของ rng เปลี่ยน)

    ─── 2. h_cls (768 dim) ───
      เป็น "summary vector" ของทั้งประโยค
      ทำไม [CLS] = summary: ดูข้อ (tt)
      ใช้ 768 dim เพราะ bert-base default; bert-large ใช้ 1024

    ─── 3. ClassificationHead ───
      Linear projection: 768 → 3 (n_classes)
      logits = h_cls @ W.T + b
        W shape = (3, 768)
        b shape = (3,)
      Dropout (0.1): ทำงานเฉพาะ training=True (ดูข้อ uu)

      Source code:
        s = sqrt(2.0 / (768 + 3))  ← He initialization
        W = randn(3, 768) * s
        b = zeros(3)

    ─── 4. Softmax ───
      แปลง logits (real numbers) → probabilities (sum=1)
      formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)
      Source ใช้ "subtract max" trick เพื่อ numerical stability:
        e = exp(logits - logits.max(axis=-1, keepdims=True))
        prob = e / e.sum(axis=-1, keepdims=True)

======================================================================
  (tt) [CLS] vs mean pooling
======================================================================

  ▶ ทำไม BERT ใช้ [CLS]:

    1. Self-attention ทำให้ [CLS] เห็นทุก token:
       [CLS] อยู่ตำแหน่งที่ 0 → ใน self-attention มันสามารถ attend ไปยัง
       ทุก token (รวม [SEP], [PAD] ที่ถูก mask)
       → หลัง 12 layers, h_[CLS] เป็น "weighted aggregation" ของทุก token

    2. Pre-trained as classification anchor:
       BERT pre-train ด้วย NSP (Next Sentence Prediction)
       ใช้ [CLS] → linear → 2-class (is next sentence?)
       → [CLS] embedding เรียนรู้แล้วว่าต้อง "encode global context"
       → fine-tune downstream task ได้ทันทีโดย swap classification head

    3. ไม่ต้อง pool/aggregate เพิ่ม:
       Output ของ encoder = (batch, seq_len, hidden)
       ใช้ [CLS] = แค่ slice [:, 0, :] → ได้ (batch, hidden) ทันที
       Simple & efficient

  ▶ เปรียบเทียบกับ MiniBERT ใน W3 (mean pooling):

    Source W3 — MiniBERT.forward (line 217):
      ctx = X.mean(axis=0)              ← เฉลี่ยทุก token
      probs = softmax(W_out @ ctx + b)

    Source W4 — BERTForClassification:
      h_cls = encoder(input_ids, mask)  ← เอาแค่ [CLS]
      probs = softmax(h_cls @ W.T + b)

    เปรียบเทียบ:

    ┌───────────────────┬──────────────────┬──────────────────┐
    │ Aspect             │ MiniBERT (W3)     │ BERT (W4)         │
    ├───────────────────┼──────────────────┼──────────────────┤
    │ Pooling             │ mean(axis=0)      │ [CLS] token       │
    │ Use of special tok  │ ไม่มี [CLS]        │ มี [CLS]           │
    │ Pre-training        │ ไม่ได้ pretrain    │ pretrain แล้ว      │
    │ Equal weight tokens│ ✓ ทุก token เท่ากัน │ ✗ [CLS] aggregate  │
    │                     │                   │   ผ่าน attention    │
    │ Sensitivity to PAD  │ สูง (ถ้าไม่ mask)  │ ต่ำ (mask อยู่ใน atten) │
    └───────────────────┴──────────────────┴──────────────────┘

  ▶ ทำไม [CLS] ดีกว่า mean pooling (ในเชิง learned):
    Mean pooling: ทุก token น้ำหนักเท่ากัน (รวม PAD, stopwords)
    [CLS]: เรียนรู้ว่า "ควรให้น้ำหนักไหนกับ token ไหน" ตอน pre-train

  ▶ แต่บางงาน mean pooling ดีกว่า:
    Sentence-BERT (สำหรับ similarity) ใช้ mean pooling
    เพราะ [CLS] ไม่ได้ optimize สำหรับ similarity task
    → ตอบคำถาม: ใช้ [CLS] หรือ mean ขึ้นอยู่กับ downstream task

======================================================================
  (uu) Dropout: training=True vs training=False
======================================================================

  ▶ Source code (ClassificationHead.forward):

    def forward(self, h_cls, training=False):
        if training:
            mask = (np.random.rand(*h_cls.shape) > self.dropout).astype(np.float32)
            h_cls = h_cls * mask / (1 - self.dropout)
        logits = h_cls @ self.W.T + self.b
        ...

  ▶ พฤติกรรม:

    ─── training=True (ตอน train) ───
      1. สุ่ม binary mask: 1 ด้วย probability (1-dropout), 0 ด้วย dropout
      2. h_cls *= mask  → ค่า ~10% ถูกเซ็ตเป็น 0 (drop)
      3. หาร (1 - dropout) → "scale up" เพื่อชดเชยค่าที่หายไป
         (เรียก inverted dropout — keep expected value คงที่)

    ─── training=False (ตอน inference) ───
      1. ไม่ทำอะไร — ใช้ h_cls เต็ม ๆ
      2. ไม่มี randomness → ผลลัพธ์ deterministic

  ▶ Demo เปรียบเทียบ:

  h_cls            = [1. 2. 3. 4. 5.]
  training=False:
    h_cls (unchanged) = [1. 2. 3. 4. 5.]
    prob               = [0.625 0.375 0.   ]

  training=True (3 ครั้ง — ดูความแตกต่าง):
    Trial 1: h_cls (dropped) = [1.67 3.33 5.   6.67 8.33]
    Trial 2: h_cls (dropped) = [0.   3.33 5.   6.67 8.33]
    Trial 3: h_cls (dropped) = [1.67 3.33 5.   0.   0.  ]

  ▶ ทำไมต้องใช้ Dropout:

    1. ป้องกัน Overfitting:
       Network จะไม่ "พึ่งพา" feature ใด feature หนึ่งมากเกินไป
       เพราะ feature นั้นอาจถูก drop ใน next batch

    2. Implicit ensemble:
       แต่ละ training step = train sub-network ที่ random
       → final model = ensemble ของ sub-networks
       → robust กว่า single network

    3. ทำไมต้อง scale ด้วย 1/(1-dropout):
       ตอน train: expected value = h_cls * (1 - dropout)
       ตอน eval:  ใช้ h_cls เต็ม (1.0×)
       → mismatch! แก้โดย scale up ตอน train ให้ expected value คงที่
       → ตอน eval ไม่ต้องทำอะไรเพิ่ม

  ▶ ใน source ใช้ dropout=0.1 (10%)
    เป็นค่ามาตรฐานของ BERT — สูงไป (0.5) จะ underfit, ต่ำไป (0.0) overfit

======================================================================
  (vv) W2 BiLSTM vs W4 BERT — และ MockBERTEncoder ขาดอะไร
======================================================================

  ▶ เปรียบเทียบ architecture:

    ┌────────────────────┬──────────────────────┬─────────────────────┐
    │ Aspect              │ W2 BiLSTM             │ W4 BERT              │
    ├────────────────────┼──────────────────────┼─────────────────────┤
    │ Sequence model      │ Recurrent (เรียงลำดับ) │ Self-attention       │
    │ Long-range dep.     │ ปานกลาง (vanishing)    │ ดี (direct attention) │
    │ Parallelization     │ ❌ sequential          │ ✅ all positions     │
    │ Pre-training        │ ❌ train from scratch  │ ✅ pretrain ใน corpus│
    │                     │                       │   ขนาดใหญ่           │
    │ Bidirectional       │ ✅ (forward+backward)  │ ✅ (full attention)  │
    │ Param count         │ ~50K (เล็ก)            │ 110M (ใหญ่)          │
    │ Train time           │ เร็ว                   │ ช้า                  │
    │ Inference            │ เร็ว                   │ ช้า                  │
    │ Domain knowledge     │ เริ่มจาก 0             │ มี world knowledge    │
    └────────────────────┴──────────────────────┴─────────────────────┘

  ▶ BERT ได้เปรียบ BiLSTM ในด้าน:

    1. Pre-trained representation:
       BERT รู้ภาษาแล้ว (syntax, semantics, world knowledge)
       BiLSTM ต้องเรียนทุกอย่างจาก dataset 66 ตัวอย่าง → underfit

    2. Long-range dependency:
       BiLSTM 'จำ' ผ่าน hidden state ที่ค่อย ๆ จาง
       BERT attention เชื่อมตรงระหว่างคู่ใด ๆ ในประโยค → ไม่จาง

    3. Contextualized embedding:
       BiLSTM: embedding ของ 'ละเมิด' = vector เดียวเสมอ
       BERT: embedding ของ 'ละเมิด' เปลี่ยนตาม context รอบข้าง
             ('ละเมิดสิทธิบัตร' vs 'ไม่ละเมิด')

    4. Transfer learning:
       Fine-tune BERT บน 66 ตัวอย่าง → ใช้ได้
       Train BiLSTM from scratch บน 66 ตัวอย่าง → overfit

  ▶ MockBERTEncoder ขาดอะไรจาก BERT จริง:

    Source: h_cls = self.rng.randn(batch, 768) * 0.1
            ← random! ไม่ใช่ encoding จริง

    ขาดสิ่งเหล่านี้:

    1. Token Embeddings (30K vocab × 768 dim)
       BERT จริง: lookup vocab → embedding ที่เรียนรู้
       Mock: ไม่มี vocab จริง

    2. Position Embeddings (max 512 × 768)
       BERT จริง: เพิ่ม positional info
       Mock: ไม่สนใจ position

    3. Segment Embeddings (2 × 768)
       BERT จริง: บอก "อยู่ประโยคแรกหรือสอง"
       Mock: ไม่รองรับ

    4. 12 Transformer Layers
       BERT จริง: 12 × (Multi-head attention + FFN + LN)
       Mock: ข้ามทั้งหมด

    5. Pre-training knowledge
       BERT จริง: train บน 16GB ของ Wikipedia + BookCorpus
       Mock: random number generator

    6. Fine-tuning capability
       BERT จริง: gradient flows → weights update
       Mock: ไม่มี gradient → no learning

  ▶ ทำไมยังใช้ MockBERTEncoder:
    - ทดสอบ pipeline / API design ก่อน install transformers
    - ใน CI/CD / unit test (เร็วและไม่ต้องโหลด model)
    - สอน concept โดยไม่ต้องรอ download model จริง

    ⚠️  ผลที่ได้จาก mock จึงไม่ meaningful — confidence ที่เห็นคือ
        random distribution ของ softmax(random vector @ random matrix)

======================================================================
  (ww) Production Roadmap — นำ W4 ไปใช้จริงในศาล
======================================================================

  ▶ 6 ขั้นตอน (เกินขั้นต่ำ 3 ตามโจทย์):

    ─── ขั้น 1: เปลี่ยน MockBERTEncoder → WangchanBERTa ───────────────
      • โหลด airesearch/wangchanberta-base-att-spm-uncased
      • ติดตั้ง transformers, torch
      • ทดสอบ tokenizer + inference บน sample จริง
      • Benchmark latency (target < 200ms/case)
      เหตุผล: ตามที่ W4.1 (nn) อธิบาย — Thai coverage 100% > 1%

    ─── ขั้น 2: รวบรวม Gold Standard Dataset (≥ 1,000 cases) ────────
      • ปัจจุบัน 66 ตัวอย่าง = "Bronze" (ไม่พอ train)
      • ขอความร่วมมือศาลทรัพย์สินทางปัญญา → ขอคำพิพากษาจริง
      • Annotator คือนักกฎหมาย IP (ไม่ใช่นักศึกษา) — Inter-annotator
        agreement ต้อง κ ≥ 0.8
      • Split: 70% train / 15% val / 15% test
      • Stratify by class + by court level (ศาลชั้นต้น/อุทธรณ์/ฎีกา)

    ─── ขั้น 3: Warm-up + Fine-tune (ตาม W4.2 pp) ──────────────────
      • Phase 1: Freeze BERT, train embedding ของ legal terms ใหม่ 3 epochs
      • Phase 2: Unfreeze ทุก layer, LR=2e-5, 5-10 epochs
      • Phase 3: Early stopping ตาม val loss (patience=3)
      • Cross-validation 5-fold → estimate generalization

    ─── ขั้น 4: Calibration + Confidence Threshold ────────────────
      • Temperature scaling: ปรับ softmax ให้ confidence ตรงความเป็นจริง
      • Set threshold: cases ที่ confidence < 0.85 → ให้นักกฎหมาย review
        (ไม่ auto-decide)
      • Add abstention class: model ตอบ "ไม่แน่ใจ" ได้

    ─── ขั้น 5: Explainability (XAI) — Required by Law ─────────────
      • Attention visualization: แสดง token ที่ model พิจารณา
      • SHAP / LIME: counterfactual explanation
        ("ถ้าไม่มีคำว่า 'ละเมิด' → prediction เปลี่ยนเป็น...")
      • Cite legal precedents ที่ใกล้เคียง (retrieval-augmented)
      • รายงาน confidence interval ไม่ใช่จุดเดียว
      เหตุผล: กฎหมายต้องการ "อธิบายได้" — black-box decision = invalid

    ─── ขั้น 6: Audit + Bias Testing ─────────────────────────────
      • Test fairness across:
         - บริษัทใหญ่ vs บริษัทเล็ก (defendant size)
         - บริษัทไทย vs ต่างชาติ (nationality bias)
         - คดีเก่า vs คดีใหม่ (temporal bias)
      • Adversarial robustness: ทดสอบ paraphrase / typo
      • Audit logs: log ทุก prediction + reasoning เพื่อ accountability
      • Periodic retraining: ทุก 6 เดือนด้วย data ใหม่

  ▶ Deployment architecture:

      Lawyer/Clerk Input
            │
            ▼
      [Pre-screening BERT]  ← W4 model
            │
       confidence ≥ 0.85?
       ┌────┴────┐
       │         │
       ▼         ▼
    Auto      Human Review
    Categorize  by IP Lawyer
       │         │
       └────┬────┘
            ▼
       Audit Log
       Final Decision

  ▶ ⚠️  สิ่งที่ห้ามทำ:
    ❌ ใช้ model auto-tัdsin คดี (decision making)
    ❌ ปกปิด: เปิดเผยว่าใช้ AI ทุกครั้ง
    ❌ ลืม update: data drift ทำให้ accuracy ตก
    ❌ ใช้ MockBERTEncoder ใน production!!! (mock = random)

"""
