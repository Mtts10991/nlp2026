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
MultiHeadAttentionSimple = w3_src.MultiHeadAttentionSimple

# ===== โค้ดบังคับตามโจทย์ =====
np.random.seed(42)
tokens = ["จำเลย", "ผลิต", "สินค้า", "ละเมิด", "สิทธิบัตร"]
X_input = np.random.randn(5, 16)            # (seq=5, d_model=16)
mha = MultiHeadAttentionSimple(d_model=16, n_heads=4, seed=42)
output, weights = mha.forward(X_input, return_weights=True)

print("=" * 70)
print("  Multi-Head Attention — Shape Analysis")
print("=" * 70)
print(f"Input:   {X_input.shape}")
print(f"Output:  {output.shape}")
print(f"Weights: {weights.shape}")

# ================================================================
# (z) อธิบาย weights shape (n_heads, seq, seq) — แต่ละ dimension หมายถึงอะไร
# W 3.2 : Z อธิบาย weights shape (n_heads, seq, seq) — แต่ละ dimension หมายถึงอะไร
# ================================================================
print("\n" + "=" * 70)
print("  (z) weights shape (n_heads, seq, seq)")
print("=" * 70)

print(f"""
  weights.shape = {weights.shape} = (n_heads, seq, seq) = (4, 5, 5)

  ▶ ความหมายแต่ละมิติ:

    dim 0 = n_heads (4)
       มี attention head 4 ตัว ทำงานขนานกัน
       แต่ละ head เรียนรู้ pattern ความสัมพันธ์ที่ต่างกัน
       เช่น head 1 อาจจับ syntax, head 2 จับ semantics

    dim 1 = seq (5) = "Query position" (i)
       token ที่กำลัง 'ถาม' ว่า — "ฉันควรสนใจ token ไหนใน sequence?"

    dim 2 = seq (5) = "Key position" (j)
       token ที่ถูก 'มองดู' โดย Query — "ฉันมีข้อมูลอะไรให้ Q บ้าง"

  ▶ weights[h, i, j] = "ใน head h, token i ให้ความสำคัญกับ token j แค่ไหน"
    (ค่าระหว่าง 0–1 ผ่าน softmax — แต่ละแถว i รวมกันได้ 1)

  ▶ ตัวอย่างจาก output จริง:
""")

"""
======================================================================
  (z) weights shape (n_heads, seq, seq)
======================================================================

  weights.shape = (4, 5, 5) = (n_heads, seq, seq) = (4, 5, 5)

  ▶ ความหมายแต่ละมิติ:

    dim 0 = n_heads (4)
       มี attention head 4 ตัว ทำงานขนานกัน
       แต่ละ head เรียนรู้ pattern ความสัมพันธ์ที่ต่างกัน
       เช่น head 1 อาจจับ syntax, head 2 จับ semantics

    dim 1 = seq (5) = "Query position" (i)
       token ที่กำลัง 'ถาม' ว่า — "ฉันควรสนใจ token ไหนใน sequence?"

    dim 2 = seq (5) = "Key position" (j)
       token ที่ถูก 'มองดู' โดย Query — "ฉันมีข้อมูลอะไรให้ Q บ้าง"

  ▶ weights[h, i, j] = "ใน head h, token i ให้ความสำคัญกับ token j แค่ไหน"
    (ค่าระหว่าง 0–1 ผ่าน softmax — แต่ละแถว i รวมกันได้ 1)

  ▶ ตัวอย่างจาก output จริง:

    Head 0 — attention pattern:
    Q\K         จำเลย      ผลิต       สินค้า     ละเมิด     สิทธิบัตร  
    จำเลย           +0.229     +0.165     +0.188     +0.215     +0.203 
    ผลิต            +0.171     +0.224     +0.221     +0.166     +0.219 
    สินค้า          +0.180     +0.225     +0.216     +0.168     +0.211 
    ละเมิด          +0.194     +0.173     +0.197     +0.227     +0.209 
    สิทธิบัตร       +0.188     +0.233     +0.216     +0.158     +0.205 

  → แถวที่ i รวมกันได้ ≈ 1 (softmax constraint):
    Head 0 row sums: ['1.000', '1.000', '1.000', '1.000', '1.000']
"""

print(f"    Head 0 — attention pattern:")
print(f"    {'Q\\K':<12}" + "".join(f"{tok:<11}" for tok in tokens))
for i, tok_q in enumerate(tokens):
    row = f"    {tok_q:<12}"
    for j in range(len(tokens)):
        row += f"{weights[0, i, j]:>+10.3f} "
    print(row)

print(f"""
  → แถวที่ i รวมกันได้ ≈ 1 (softmax constraint):
    Head 0 row sums: {[f'{weights[0, i].sum():.3f}' for i in range(5)]}
""")

# ================================================================
# (aa) อธิบายทุกขั้นตอนใน forward(): layer_norm → Q,K,V → split_heads → attention → combine → residual
# W 3.2 : AA อธิบายทุกขั้นตอนใน forward(): layer_norm → Q,K,V projection → split_heads → attention → combine → residual
# ================================================================
print("=" * 70)
print("  (aa) Step-by-step ใน mha.forward()")
print("=" * 70)

# Manual replay เพื่อ trace shape แต่ละ step
x = X_input[np.newaxis, :]                                       # add batch dim
x_norm = mha.layer_norm(x)                                       # Pre-Norm
Q_full = np.matmul(x_norm, mha.W_q)
Q = mha.split_heads(Q_full)                                      # split heads
batch, seq, _ = x.shape
Q_r = Q.reshape(batch * mha.n_heads, seq, mha.d_k)               # flatten heads → batch
attn_w = weights[np.newaxis, :]                                  # already from forward
combined = mha.combine_heads(np.zeros((1, mha.n_heads, seq, mha.d_k)))   # demo shape

print(f"""
  Input X: (1, 5, 16) ← (batch, seq, d_model)
    │  ── batch dim ถูกเพิ่มถ้า input เป็น 2D
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 1: layer_norm(x)  — Pre-Norm                                  │
  │   normalize ต่อ feature dim: (x - mean) / (std + eps)              │
  │   shape: {x_norm.shape}  ← unchanged                             │
  │   เหตุผล: stabilize input distribution ก่อนเข้า attention            │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 2: Q, K, V projection                                          │
  │   Q = x_norm @ W_q  shape: {Q_full.shape}                        │
  │   K = x_norm @ W_k                                                  │
  │   V = x_norm @ W_v                                                  │
  │   เปลี่ยน "embedding" → 3 มุมมอง:                                   │
  │     Q (Query) — ฉันถามอะไร?                                          │
  │     K (Key)   — ฉันมีอะไรให้ค้นหา?                                   │
  │     V (Value) — ข้อมูลจริงคืออะไร?                                   │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 3: split_heads — แบ่ง d_model = 16 เป็น 4 heads × 4 dim       │
  │   reshape: (1, 5, 16) → (1, 5, 4, 4)                                │
  │   transpose: → {Q.shape}  (batch, n_heads, seq, d_k)            │
  │   แต่ละ head เห็น 4 dim ของ embedding → คิดสัมพันธ์ต่าง pattern        │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 4: scaled_dot_product_attention                                │
  │   reshape: (1, 4, 5, 4) → (4, 5, 4) ← merge batch+heads             │
  │   scores = Q @ K.T / sqrt(d_k)         shape: (4, 5, 5)             │
  │   weights = softmax(scores)             ← row-wise sum=1            │
  │   attn_out = weights @ V                shape: (4, 5, 4)            │
  │   ความหมาย: แต่ละ token i ได้ output = ผลรวมถ่วงน้ำหนักของทุก V      │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 5: combine_heads — รวม heads กลับเป็น d_model                  │
  │   reshape: (4, 5, 4) → (1, 4, 5, 4)                                 │
  │   transpose: → (1, 5, 4, 4) → reshape (1, 5, 16)                    │
  │   = แต่ละ token มี 16 dim ที่เป็น concatenation ของ 4 heads          │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 6: output projection — W_o                                     │
  │   output = attn_out @ W_o      shape: (1, 5, 16)                    │
  │   ผสม info จากทุก head เข้าด้วยกันก่อนส่งต่อ                            │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 7: residual connection                                          │
  │   output = output + x   ← บวก x เดิม (ไม่ใช่ x_norm!)                 │
  │   ป้องกัน vanishing gradient ใน deep network                          │
  │   ทำให้ network เรียนรู้ "delta" แทนที่จะ "absolute"                   │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  Final output: {output.shape}  ← เท่ากับ input shape (5, 16)
""")

"""
======================================================================
  (aa) Step-by-step ใน mha.forward()
======================================================================

  Input X: (1, 5, 16) ← (batch, seq, d_model)
    │  ── batch dim ถูกเพิ่มถ้า input เป็น 2D
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 1: layer_norm(x)  — Pre-Norm                                  │
  │   normalize ต่อ feature dim: (x - mean) / (std + eps)              │
  │   shape: (1, 5, 16)  ← unchanged                             │
  │   เหตุผล: stabilize input distribution ก่อนเข้า attention            │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 2: Q, K, V projection                                          │
  │   Q = x_norm @ W_q  shape: (1, 5, 16)                        │
  │   K = x_norm @ W_k                                                  │
  │   V = x_norm @ W_v                                                  │
  │   เปลี่ยน "embedding" → 3 มุมมอง:                                   │
  │     Q (Query) — ฉันถามอะไร?                                          │
  │     K (Key)   — ฉันมีอะไรให้ค้นหา?                                   │
  │     V (Value) — ข้อมูลจริงคืออะไร?                                   │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 3: split_heads — แบ่ง d_model = 16 เป็น 4 heads × 4 dim       │
  │   reshape: (1, 5, 16) → (1, 5, 4, 4)                                │
  │   transpose: → (1, 4, 5, 4)  (batch, n_heads, seq, d_k)            │
  │   แต่ละ head เห็น 4 dim ของ embedding → คิดสัมพันธ์ต่าง pattern        │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 4: scaled_dot_product_attention                                │
  │   reshape: (1, 4, 5, 4) → (4, 5, 4) ← merge batch+heads             │
  │   scores = Q @ K.T / sqrt(d_k)         shape: (4, 5, 5)             │
  │   weights = softmax(scores)             ← row-wise sum=1            │
  │   attn_out = weights @ V                shape: (4, 5, 4)            │
  │   ความหมาย: แต่ละ token i ได้ output = ผลรวมถ่วงน้ำหนักของทุก V      │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 5: combine_heads — รวม heads กลับเป็น d_model                  │
  │   reshape: (4, 5, 4) → (1, 4, 5, 4)                                 │
  │   transpose: → (1, 5, 4, 4) → reshape (1, 5, 16)                    │
  │   = แต่ละ token มี 16 dim ที่เป็น concatenation ของ 4 heads          │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 6: output projection — W_o                                     │
  │   output = attn_out @ W_o      shape: (1, 5, 16)                    │
  │   ผสม info จากทุก head เข้าด้วยกันก่อนส่งต่อ                            │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 7: residual connection                                          │
  │   output = output + x   ← บวก x เดิม (ไม่ใช่ x_norm!)                 │
  │   ป้องกัน vanishing gradient ใน deep network                          │
  │   ทำให้ network เรียนรู้ "delta" แทนที่จะ "absolute"                   │
  └───────────────────────────────────────────────────────────────────┘
    │
    ▼
  Final output: (5, 16)  ← เท่ากับ input shape (5, 16)
"""

# ================================================================
# (bb) ทำไม d_k = d_model // n_heads = 4 — ถ้า n_heads=8 จะเกิดอะไรกับ d_k
# W 3.2 : BB เหตุใด d_k = d_model // n_heads = 4 — ถ้า n_heads=8 จะเกิดอะไรกับ d_k
# ================================================================
print("=" * 70)
print("  (bb) d_k = d_model // n_heads")
print("=" * 70)

print(f"""
  สูตร: d_k = d_model // n_heads
        d_v = d_model // n_heads (มักเท่ากัน)

  ใน MHA นี้:
    d_model = 16, n_heads = 4 → d_k = 16 // 4 = {mha.d_k}

  ▶ ทำไมต้องหาร:
    1. คงต้นทุน computation: ทุก head รวมกันมี dim ทั้งหมด = d_model
       ไม่ว่าจะแบ่งเป็นกี่ head, จำนวน parameter ของ Q,K,V projection คงเดิม
       (W_q เป็น d_model × d_model = 16 × 16 = 256 params ไม่เปลี่ยน)

    2. แต่ละ head เรียนรู้ pattern ในมุมมองที่แคบลง (sub-space)
       4 heads × 4 dim = 16 dim รวม ≈ 1 head × 16 dim แต่ "หลายมุม" กว่า

  ▶ ถ้า n_heads = 8:
""")

# ทดลองสร้าง MHA n_heads=8
mha8 = MultiHeadAttentionSimple(d_model=16, n_heads=8, seed=42)
print(f"    d_model = 16, n_heads = 8 → d_k = 16 // 8 = {mha8.d_k}")
out8, w8 = mha8.forward(X_input, return_weights=True)
print(f"    weights.shape: {w8.shape}  (n_heads, seq, seq)")
print(f"    output.shape : {out8.shape}")

print(f"""
  ▶ ผลของ d_k ที่เล็กลง:
    - แต่ละ head ทำงานใน 2-D sub-space เท่านั้น (น้อยมาก!)
    - attention score scaled ด้วย sqrt(d_k) = sqrt(2) ≈ 1.41
      แทน sqrt(4) = 2 → softmax อาจ peak มากขึ้น
    - ใน practice: ถ้า d_k ต่ำเกิน → head แต่ละตัวจับ pattern ได้แคบ
      Standard rule: d_k ควร ≥ 32 หรือ 64 ใน production model

  ▶ ถ้า d_model หารด้วย n_heads ไม่ลงตัว:
    เช่น d_model=16, n_heads=3 → 16 // 3 = 5  (เหลือ dim ไม่ใช้!)
    split_heads จะ error เพราะ reshape ขนาดไม่ตรง
    → ต้องเลือก n_heads ที่หาร d_model ลงตัวเสมอ
""")

"""
======================================================================
  (bb) d_k = d_model // n_heads
======================================================================

  สูตร: d_k = d_model // n_heads
        d_v = d_model // n_heads (มักเท่ากัน)

  ใน MHA นี้:
    d_model = 16, n_heads = 4 → d_k = 16 // 4 = 4

  ▶ ทำไมต้องหาร:
    1. คงต้นทุน computation: ทุก head รวมกันมี dim ทั้งหมด = d_model
       ไม่ว่าจะแบ่งเป็นกี่ head, จำนวน parameter ของ Q,K,V projection คงเดิม
       (W_q เป็น d_model × d_model = 16 × 16 = 256 params ไม่เปลี่ยน)

    2. แต่ละ head เรียนรู้ pattern ในมุมมองที่แคบลง (sub-space)
       4 heads × 4 dim = 16 dim รวม ≈ 1 head × 16 dim แต่ "หลายมุม" กว่า

  ▶ ถ้า n_heads = 8:

    d_model = 16, n_heads = 8 → d_k = 16 // 8 = 2
    weights.shape: (8, 5, 5)  (n_heads, seq, seq)
    output.shape : (5, 16)

  ▶ ผลของ d_k ที่เล็กลง:
    - แต่ละ head ทำงานใน 2-D sub-space เท่านั้น (น้อยมาก!)
    - attention score scaled ด้วย sqrt(d_k) = sqrt(2) ≈ 1.41
      แทน sqrt(4) = 2 → softmax อาจ peak มากขึ้น
    - ใน practice: ถ้า d_k ต่ำเกิน → head แต่ละตัวจับ pattern ได้แคบ
      Standard rule: d_k ควร ≥ 32 หรือ 64 ใน production model

  ▶ ถ้า d_model หารด้วย n_heads ไม่ลงตัว:
    เช่น d_model=16, n_heads=3 → 16 // 3 = 5  (เหลือ dim ไม่ใช้!)
    split_heads จะ error เพราะ reshape ขนาดไม่ตรง
    → ต้องเลือก n_heads ที่หาร d_model ลงตัวเสมอ
"""

# ================================================================
# (cc) Pre-Norm คืออะไร — ต่างจาก Post-Norm อย่างไร และทำไม gradient ไหลดีกว่า
# W 3.2 : CC Pre-Norm คืออะไร — แตกต่างจาก Post-Norm อย่างไร และทำไม gradient ไหลดีกว่า
# ================================================================
print("=" * 70)
print("  (cc) Pre-Norm vs Post-Norm")
print("=" * 70)

print(f"""
  ▶ Post-Norm (Original Transformer 2017):
      x_out = LayerNorm(x + Sublayer(x))
                ↑ normalize หลัง attention + residual

      pseudo:
        attn = Attention(x)
        x = x + attn          # residual
        x = LayerNorm(x)      # normalize ทีหลัง

  ▶ Pre-Norm (ใช้ใน source / GPT-2/3, BERT v2):
      x_out = x + Sublayer(LayerNorm(x))
                ↑ normalize ก่อนเข้า attention

      pseudo (ตรงกับ source):
        x_norm = LayerNorm(x)
        attn = Attention(x_norm)
        x = x + attn          # residual บวกกับ x ดิบ ไม่ใช่ x_norm

  ▶ ความต่างสำคัญ:

    ┌──────────────────────┬──────────────────┬──────────────────┐
    │ Property              │ Post-Norm         │ Pre-Norm         │
    ├──────────────────────┼──────────────────┼──────────────────┤
    │ Stability              │ ยาก (ต้อง warmup) │ ง่าย              │
    │ Gradient flow          │ อ่อนใน deep net   │ คงตัวข้าม layer   │
    │ Final layer norm       │ ไม่จำเป็น          │ ต้องใส่หลัง stack  │
    │ Convergence speed      │ ช้า               │ เร็วกว่า           │
    │ Deep network (12+ ชั้น) │ พังบ่อย            │ stable            │
    └──────────────────────┴──────────────────┴──────────────────┘

  ▶ ทำไม Pre-Norm gradient ไหลดีกว่า:

    1. Identity path "บริสุทธิ์":
       Pre-Norm: x_out = x + Sublayer(LN(x))
       gradient ของ x_out ต่อ x = 1 + ∂Sublayer/∂x  ← มี "1" ตรงๆ
       → gradient flow ผ่าน residual path โดยไม่ถูก LN กดทับ

       Post-Norm: x_out = LN(x + Sublayer(x))
       gradient = ∂LN/∂(...) × (1 + ∂Sublayer/∂x)
       → ถูก factor ของ LN (ที่อาจเล็ก) คูณ → vanish เมื่อ stack หลายชั้น

    2. Identity gradient ใน deep stack:
       12-layer Pre-Norm: gradient ที่ layer 1 ≈ Σ gradient จากทุก layer
       12-layer Post-Norm: gradient ที่ layer 1 ≈ ∏ (LN factors) → 0

    3. ไม่ต้อง learning rate warmup:
       Post-Norm ต้อง warmup หลายพัน step ไม่งั้น loss explode
       Pre-Norm train ได้ตั้งแต่ step แรก

  ▶ Trade-off:
    Pre-Norm ทำงาน stable กว่ามาก แต่ "perform" ได้ค่อนข้างต่ำกว่า
    Post-Norm ที่ทำให้ทำงานได้ — เลยมี variants ใหม่ ๆ เช่น
    Sandwich-Norm, DeepNorm ที่พยายามรวมข้อดีของทั้งสอง

  ▶ ใน source's MultiHeadAttentionSimple:
    บรรทัด 80: x_norm = self.layer_norm(x)         ← LN ก่อน
    บรรทัด 111: output = output + x                  ← residual กับ x ดิบ
    = Pre-Norm pattern ครบสูตร
""")
