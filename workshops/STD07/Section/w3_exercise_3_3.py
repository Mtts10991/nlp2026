import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import importlib.util
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

# ===== Import จาก w3 source (มี space ในชื่อ) =====
SRC_PATH = Path(__file__).parent / "w3 Attention_legal.py"
spec = importlib.util.spec_from_file_location("w3_src", SRC_PATH)
w3_src = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()):
    spec.loader.exec_module(w3_src)

TransformerEncoderBlock = w3_src.TransformerEncoderBlock
TransformerDecoderBlock = w3_src.TransformerDecoderBlock
_print_heatmap = w3_src._print_heatmap

# ===== โค้ดบังคับตามโจทย์ =====
d_model, n_heads, seq_len = 32, 4, 5
rng = np.random.RandomState(0)
X_emb = rng.randn(seq_len, d_model)
enc = TransformerEncoderBlock(d_model, n_heads)
dec = TransformerDecoderBlock(d_model, n_heads)
enc_out, enc_ws = enc.forward(X_emb, return_weights=True)
dec_out, dec_ws = dec.forward(X_emb, return_weights=True)

print("=" * 70)
print("  Encoder vs Decoder — Heatmap (Head 1)")
print("=" * 70)
print("\n  ── Encoder Block (Full Attention) ──")
_print_heatmap(enc_ws[0], seq_len)
print("\n  ── Decoder Block (Causal Mask) ──")
_print_heatmap(dec_ws[0], seq_len)

# ================================================================
# (dd) อธิบายความแตกต่างของ heatmap — token t3 มองเห็น token ใดบ้างในแต่ละแบบ
# W 3.3 : DD อธิบายความแตกต่างของ heatmap ระหว่าง Encoder กับ Decoder — token t3 มองเห็น token ใดได้บ้างในแต่ละแบบ
# ================================================================
print("\n" + "=" * 70)
print("  (dd) Heatmap Encoder vs Decoder — token t3 มองเห็นใครบ้าง")
print("=" * 70)

print(f"""
  ▶ จาก heatmap ด้านบน:

    Encoder (Full Attention):
      t3 row → ให้ความสำคัญกับ t0, t1, t2, t3, t4 (ทุก token)
      = "อ่านได้ทั้งประโยค" → bidirectional context

    Decoder (Causal Mask):
      t3 row → ให้ความสำคัญกับ t0, t1, t2, t3 (เท่านั้น) — t4 = 0
      = "อ่านได้แค่อดีต + ตัวเอง" → unidirectional (left-to-right)

  ▶ ค่า attention weight ของ t3 (head 1):

    Position    Encoder     Decoder
    ────────    ───────     ───────""")
for j in range(seq_len):
    enc_w = enc_ws[0][3, j]
    dec_w = dec_ws[0][3, j]
    masked = " ← MASKED" if dec_w < 1e-8 else ""
    print(f"    t3 → t{j}    {enc_w:.4f}     {dec_w:.4f}{masked}")

print(f"""
  ▶ ผลกระทบใน NLP:

    Encoder ใช้ใน task ที่ต้องเข้าใจประโยคทั้งหมด:
      - Classification (เช่น คดีนี้เป็น patent หรือ copyright?)
      - Named Entity Recognition (มี 'มาตรา 77' ในประโยคไหน?)
      - Sentence embedding (vector ของ "ทั้งประโยค")

    Decoder ใช้ใน task ที่ต้องสร้าง token ทีละตัว:
      - Text generation (สร้างคำพิพากษา, สรุปคดี)
      - Translation (ภาษาอังกฤษ → ไทย)
      - Auto-completion (พิมพ์คำแรก → ทาย next token)

    ทำไม Decoder ต้อง mask:
      ตอน train: model ต้องไม่ "แอบดู" คำตอบในอนาคต ไม่งั้นจะ cheat
      เช่น สอนให้ทาย t4 → ห้าม t3 มองเห็น t4
""")

"""
======================================================================
  (dd) Heatmap Encoder vs Decoder — token t3 มองเห็นใครบ้าง
======================================================================

  ▶ จาก heatmap ด้านบน:

    Encoder (Full Attention):
      t3 row → ให้ความสำคัญกับ t0, t1, t2, t3, t4 (ทุก token)
      = "อ่านได้ทั้งประโยค" → bidirectional context

    Decoder (Causal Mask):
      t3 row → ให้ความสำคัญกับ t0, t1, t2, t3 (เท่านั้น) — t4 = 0
      = "อ่านได้แค่อดีต + ตัวเอง" → unidirectional (left-to-right)

  ▶ ค่า attention weight ของ t3 (head 1):

    Position    Encoder     Decoder
    ────────    ───────     ───────
    t3 → t0    0.2247     0.2806
    t3 → t1    0.1708     0.2134
    t3 → t2    0.2167     0.2706
    t3 → t3    0.1886     0.2355
    t3 → t4    0.1993     0.0000 ← MASKED

  ▶ ผลกระทบใน NLP:

    Encoder ใช้ใน task ที่ต้องเข้าใจประโยคทั้งหมด:
      - Classification (เช่น คดีนี้เป็น patent หรือ copyright?)
      - Named Entity Recognition (มี 'มาตรา 77' ในประโยคไหน?)
      - Sentence embedding (vector ของ "ทั้งประโยค")

    Decoder ใช้ใน task ที่ต้องสร้าง token ทีละตัว:
      - Text generation (สร้างคำพิพากษา, สรุปคดี)
      - Translation (ภาษาอังกฤษ → ไทย)
      - Auto-completion (พิมพ์คำแรก → ทาย next token)

    ทำไม Decoder ต้อง mask:
      ตอน train: model ต้องไม่ "แอบดู" คำตอบในอนาคต ไม่งั้นจะ cheat
      เช่น สอนให้ทาย t4 → ห้าม t3 มองเห็น t4
"""

# ================================================================
# (ee) Causal Mask สร้างอย่างไร — np.triu(np.ones((T,T), dtype=bool), k=1) หมายความว่าอะไร
# W 3.3 : EE Causal Mask สร้างอย่างไร — np.triu(np.ones((T,T), dtype=bool), k=1) หมายความว่าอะไร
# ================================================================
print("=" * 70)
print("  (ee) Causal Mask: np.triu(np.ones((T,T), dtype=bool), k=1)")
print("=" * 70)

T = 5
mask = np.triu(np.ones((T, T), dtype=bool), k=1)

print(f"""
  ▶ Components ของสูตร:

    1. np.ones((T,T), dtype=bool):
       สร้าง matrix T×T เต็มไปด้วย True

    2. np.triu(M, k=1):
       เก็บเฉพาะ "upper triangle" ตั้งแต่ diagonal ที่ k=1 ขึ้นไป
       (k=0 รวม diagonal หลัก, k=1 ข้าม diagonal หลัก)
""")

print(f"  ▶ ผลลัพธ์ (T={T}):")
print(f"    {'i\\j':<6}" + "".join(f"  t{j}" for j in range(T)))
for i in range(T):
    row = f"    t{i:<4}"
    for j in range(T):
        row += "   T" if mask[i, j] else "   ·"
    print(row)

print(f"""
  ▶ ความหมาย mask[i, j] = True:
    "ตำแหน่ง i ไม่อนุญาตให้ดู j" (j อยู่ในอนาคตของ i)

  ▶ ทำไม k=1 ไม่ใช่ k=0:
    k=0 จะ mask ทั้ง diagonal → token i มองตัวเองไม่ได้ (ผิด!)
    k=1 ข้าม diagonal → token i มองตัวเองได้ + อดีต (ถูก)

  ▶ ใน source's TransformerDecoderBlock:
    @staticmethod
    def _causal_mask(T):
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    ส่งไปที่ scale_dot_product_attention ที่จัดการ -1e9 ต่อ
""")

"""
======================================================================
  (ee) Causal Mask: np.triu(np.ones((T,T), dtype=bool), k=1)
======================================================================

  ▶ Components ของสูตร:

    1. np.ones((T,T), dtype=bool):
       สร้าง matrix T×T เต็มไปด้วย True

    2. np.triu(M, k=1):
       เก็บเฉพาะ "upper triangle" ตั้งแต่ diagonal ที่ k=1 ขึ้นไป
       (k=0 รวม diagonal หลัก, k=1 ข้าม diagonal หลัก)

  ▶ ผลลัพธ์ (T=5):
    i\j     t0  t1  t2  t3  t4
    t0      ·   T   T   T   T
    t1      ·   ·   T   T   T
    t2      ·   ·   ·   T   T
    t3      ·   ·   ·   ·   T
    t4      ·   ·   ·   ·   ·

  ▶ ความหมาย mask[i, j] = True:
    "ตำแหน่ง i ไม่อนุญาตให้ดู j" (j อยู่ในอนาคตของ i)

  ▶ ทำไม k=1 ไม่ใช่ k=0:
    k=0 จะ mask ทั้ง diagonal → token i มองตัวเองไม่ได้ (ผิด!)
    k=1 ข้าม diagonal → token i มองตัวเองได้ + อดีต (ถูก)

  ▶ ใน source's TransformerDecoderBlock:
    @staticmethod
    def _causal_mask(T):
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    ส่งไปที่ scale_dot_product_attention ที่จัดการ -1e9 ต่อ
"""

# ================================================================
# (ff) ค่า -1e9 ใน causal mask ทำงานร่วมกับ softmax อย่างไร — ทำไมต้องใช้ -1e9 ไม่ใช่ 0
# W 3.3 : FF ค่า -1e9 ใน causal mask ทำงานร่วมกับ softmax อย่างไร — ทำไมต้องใช้ -1e9 ไม่ใช่ 0
# ================================================================
print("=" * 70)
print("  (ff) ทำไมใช้ -1e9 (ไม่ใช่ 0) ใน causal mask")
print("=" * 70)

print(f"""
  ▶ Source code ที่ scale_dot_product_attention:
      if causal_mask is not None:
          scores = np.where(causal_mask, -1e9, scores)
      ...
      exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
      weights = exp_s / exp_s.sum(axis=-1, keepdims=True)

  ▶ Pipeline ของ softmax:
      scores → exp(scores) → normalize → weights (sum=1)

  ▶ ทำไมไม่ใช้ 0:
""")

# พิสูจน์ด้วยตัวเลขจริง
scores_demo = np.array([2.0, 1.0, 3.0, 0.5, 1.5])
mask_demo = np.array([False, False, False, True, True])  # t3, t4 = future

# ผิด: zero out scores
scores_zero = np.where(mask_demo, 0.0, scores_demo)
exp_zero = np.exp(scores_zero - np.max(scores_zero))
sm_zero = exp_zero / exp_zero.sum()

# ถูก: -inf (proxy = -1e9)
scores_neg = np.where(mask_demo, -1e9, scores_demo)
exp_neg = np.exp(scores_neg - np.max(scores_neg))
sm_neg = exp_neg / exp_neg.sum()

print(f"    Demo: scores = {scores_demo.tolist()}")
print(f"          mask   = {mask_demo.tolist()}  (True = ตำแหน่งต้อง mask)")
print(f"")
print(f"    ❌ ถ้าใช้ 0 แทน scores ที่ mask:")
print(f"       scores_zero = {scores_zero.tolist()}")
print(f"       softmax     = {[f'{v:.3f}' for v in sm_zero]}")
print(f"       → t3, t4 ยังได้ weight {sm_zero[3]:.3f}, {sm_zero[4]:.3f} > 0  (ผิด! ยังรั่ว)")
print(f"")
print(f"    ✅ ใช้ -1e9:")
print(f"       scores_neg  = [..., -1e9, -1e9]")
print(f"       softmax     = {[f'{v:.3f}' for v in sm_neg]}")
print(f"       → t3, t4 ได้ weight = 0 จริง ๆ  (ถูก!)")

print(f"""
  ▶ คณิตศาสตร์เบื้องหลัง:
    softmax(x_i) = exp(x_i) / Σ exp(x_j)

    ใช้ 0:    exp(0) = 1                → ยังมีน้ำหนักเหลือ
    ใช้ -1e9: exp(-1e9) ≈ 0 (underflow) → น้ำหนัก = 0 จริง

    หลังจาก subtract max (numerical stability):
      ใช้ 0:    score = -max < 0 แต่ exp ยังให้ค่าเล็ก ๆ ไม่เป็นศูนย์
      ใช้ -1e9: score - max ยังเป็น -1e9 → exp ใต้ floating-point precision = 0

  ▶ ทำไมไม่ใช้ -∞ (np.inf):
    -inf อาจทำให้ NaN ตอน max(-inf, ...) ในบาง numpy version
    -1e9 เป็น "ใหญ่พอที่ exp() จะได้ 0" แต่ยังเป็น finite number
    safe กว่าทุกกรณี
""")

"""
======================================================================
  (ff) ทำไมใช้ -1e9 (ไม่ใช่ 0) ใน causal mask
======================================================================

  ▶ Source code ที่ scale_dot_product_attention:
      if causal_mask is not None:
          scores = np.where(causal_mask, -1e9, scores)
      ...
      exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
      weights = exp_s / exp_s.sum(axis=-1, keepdims=True)

  ▶ Pipeline ของ softmax:
      scores → exp(scores) → normalize → weights (sum=1)

  ▶ ทำไมไม่ใช้ 0:

    Demo: scores = [2.0, 1.0, 3.0, 0.5, 1.5]
          mask   = [False, False, False, True, True]  (True = ตำแหน่งต้อง mask)

    ❌ ถ้าใช้ 0 แทน scores ที่ mask:
       scores_zero = [2.0, 1.0, 3.0, 0.0, 0.0]
       softmax     = ['0.230', '0.084', '0.624', '0.031', '0.031']
       → t3, t4 ยังได้ weight 0.031, 0.031 > 0  (ผิด! ยังรั่ว)

    ✅ ใช้ -1e9:
       scores_neg  = [..., -1e9, -1e9]
       softmax     = ['0.245', '0.090', '0.665', '0.000', '0.000']
       → t3, t4 ได้ weight = 0 จริง ๆ  (ถูก!)

  ▶ คณิตศาสตร์เบื้องหลัง:
    softmax(x_i) = exp(x_i) / Σ exp(x_j)

    ใช้ 0:    exp(0) = 1                → ยังมีน้ำหนักเหลือ
    ใช้ -1e9: exp(-1e9) ≈ 0 (underflow) → น้ำหนัก = 0 จริง

    หลังจาก subtract max (numerical stability):
      ใช้ 0:    score = -max < 0 แต่ exp ยังให้ค่าเล็ก ๆ ไม่เป็นศูนย์
      ใช้ -1e9: score - max ยังเป็น -1e9 → exp ใต้ floating-point precision = 0

  ▶ ทำไมไม่ใช้ -∞ (np.inf):
    -inf อาจทำให้ NaN ตอน max(-inf, ...) ในบาง numpy version
    -1e9 เป็น "ใหญ่พอที่ exp() จะได้ 0" แต่ยังเป็น finite number
    safe กว่าทุกกรณี
"""

# ================================================================
# (gg) Classification คดี IP ควรใช้ Encoder หรือ Decoder
# W 3.3 : GG งาน classification คดี IP ควรใช้ Encoder หรือ Decoder — อธิบายเหตุผลจาก output mechanism
# ================================================================
print("=" * 70)
print("  (gg) Classification คดี IP ควรใช้ Encoder หรือ Decoder")
print("=" * 70)

print(f"""
  ▶ คำตอบ: Encoder (เช่น MiniBERT ใน source)

  ▶ เปรียบเทียบ output mechanism จาก source:

    MiniBERT (Encoder-based) — บรรทัด 217-219:
      ctx = X.mean(axis=0)                              ← mean pool ทุก token
      probs = self._softmax(self.W_out @ ctx + b_out)   ← classify จาก ctx เดียว

      = บีบทั้งประโยค (5 token × 32 dim) → context vector เดียว (32 dim)
        แล้ว predict 1 class

    MiniGPT (Decoder-based) — บรรทัด 242:
      return self.W_lm @ X[-1], all_weights              ← เอาเฉพาะ last token

      = ใช้ token สุดท้ายของประโยค → predict next token (vocab probabilities)

  ▶ 3 เหตุผลที่ Encoder เหมาะกว่าสำหรับ classification:

    1. Bidirectional context:
       คดี IP ต้องอ่านทั้งประโยคก่อนตัดสิน เช่น
       "จำเลย___ผลิตยาที่เลียนแบบ___สิทธิบัตร___"
       Encoder อ่าน "ทุกคำทุกตำแหน่ง" → เข้าใจความสัมพันธ์ครบ
       Decoder อ่านได้แค่อดีต → ตัดสินจากครึ่งประโยค (ผิดง่าย)

    2. Output design:
       Classification = 1 input → 1 output (1 class label)
       Encoder mean-pool → vector เดียว ตรงกับ task pattern
       Decoder next-token → ออกแบบสำหรับ generation (ผิด task)

    3. Training efficiency:
       Encoder train แบบ MLM (Masked Language Model) — เห็นทั้งประโยค
       เรียนรู้ representation ที่ลึก เร็วกว่า
       Decoder train แบบ causal LM — แต่ละ token เห็น "ครึ่ง" → ใช้ data
       เยอะกว่ากว่าจะ converge

  ▶ ตัวอย่างจาก industry:

    Task                  | Architecture        | ตัวอย่าง model
    ─────────────────────|─────────────────────|──────────────
    Classification (เรา) | Encoder              | BERT, RoBERTa, WangchanBERTa
    NER                   | Encoder              | BERT-NER
    Sentence similarity   | Encoder              | Sentence-BERT
    Text generation       | Decoder              | GPT-2, GPT-4, Claude
    Translation           | Encoder + Decoder    | T5, BART (sequence-to-sequence)

  ▶ สรุป: คดี IP ของเราเป็น classification (3 class)
          → ใช้ MiniBERT (Encoder) ไม่ใช่ MiniGPT (Decoder)
""")
