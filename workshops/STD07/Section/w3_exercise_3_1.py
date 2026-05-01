import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import importlib.util
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

# ===== Import จาก w3 source (ชื่อไฟล์มี space) =====
SRC_PATH = Path(__file__).parent / "w3 Attention_legal.py"
spec = importlib.util.spec_from_file_location("w3_src", SRC_PATH)
w3_src = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()):
    spec.loader.exec_module(w3_src)
SinusodalPositionEncoding = w3_src.SinusodalPositionEncoding

# ===== โค้ดบังคับตามโจทย์ =====
pe = SinusodalPositionEncoding(max_seq_len=10, d_model=16)
words = ["จำเลย", "ผลิต", "สินค้า", "ละเมิด", "สิทธิบัตร"]

print("=" * 70)
print("  Sinusoidal Positional Encoding — show_with_words(words)")
print("=" * 70)
pe.show_with_words(words)

# ================================================================
# (w) PE ของ pos=0 กับ pos=4 ต่างกันอย่างไรใน dimension แรก — sin/cos pattern
# W 3.1 : W PE ของ pos=0 กับ pos=4 ต่างกันอย่างไรใน dimension แรก — อธิบาย pattern sin/cos
# ================================================================
print("\n" + "=" * 70)
print("  (w) PE pos=0 vs pos=4 ต่างกันอย่างไรใน dim แรก")
print("=" * 70)

pe0 = pe.pe[0]
pe4 = pe.pe[4]

print(f"\n  PE[pos=0, :8] = {[f'{v:+.3f}' for v in pe0[:8]]}")
print(f"  PE[pos=4, :8] = {[f'{v:+.3f}' for v in pe4[:8]]}")
print(f"\n  ต่างกันใน dim แรก: PE[0,0] = {pe0[0]:+.3f}  →  PE[4,0] = {pe4[0]:+.3f}")
print(f"                       Δ = {pe4[0] - pe0[0]:+.3f}")

print(f"""
  ▶ Pattern sin/cos:
    Formula ใน source:
      pe[pos, 2i  ] = sin(pos / 10000^(2i/d_model))
      pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    dim 0 (i=0): sin(pos / 10000^0) = sin(pos / 1) = sin(pos)
    dim 1 (i=0): cos(pos / 10000^0) = cos(pos / 1) = cos(pos)
    dim 2 (i=1): sin(pos / 10000^(2/16)) = sin(pos / 3.16)
    dim 3 (i=1): cos(pos / 10000^(2/16)) = cos(pos / 3.16)
    ...

    ▶ pos=0:  sin(0)=0      , cos(0)=1
    ▶ pos=4:  sin(4)≈-0.757 , cos(4)≈-0.654

    → ทุก dim มีความถี่ (frequency) ต่างกัน:
       dim ต่ำ ๆ → ความถี่สูง (เปลี่ยนเร็วตาม pos)
       dim สูง ๆ → ความถี่ต่ำ (เปลี่ยนช้า — pos ห่างกันยังคล้ายกัน)
    → ทำให้ทุก position มี "ลายนิ้วมือ" เฉพาะตัวที่โมเดลแยกแยะได้
""")

"""
======================================================================
  (w) PE pos=0 vs pos=4 ต่างกันอย่างไรใน dim แรก
======================================================================

  PE[pos=0, :8] = ['+0.000', '+1.000', '+0.000', '+1.000', '+0.000', '+1.000', '+0.000', '+1.000']
  PE[pos=4, :8] = ['-0.757', '-0.654', '+0.954', '+0.301', '+0.389', '+0.921', '+0.126', '+0.992']

  ต่างกันใน dim แรก: PE[0,0] = +0.000  →  PE[4,0] = -0.757
                       Δ = -0.757

  ▶ Pattern sin/cos:
    Formula ใน source:
      pe[pos, 2i  ] = sin(pos / 10000^(2i/d_model))
      pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    dim 0 (i=0): sin(pos / 10000^0) = sin(pos / 1) = sin(pos)
    dim 1 (i=0): cos(pos / 10000^0) = cos(pos / 1) = cos(pos)
    dim 2 (i=1): sin(pos / 10000^(2/16)) = sin(pos / 3.16)
    dim 3 (i=1): cos(pos / 10000^(2/16)) = cos(pos / 3.16)
    ...

    ▶ pos=0:  sin(0)=0      , cos(0)=1
    ▶ pos=4:  sin(4)≈-0.757 , cos(4)≈-0.654

    → ทุก dim มีความถี่ (frequency) ต่างกัน:
       dim ต่ำ ๆ → ความถี่สูง (เปลี่ยนเร็วตาม pos)
       dim สูง ๆ → ความถี่ต่ำ (เปลี่ยนช้า — pos ห่างกันยังคล้ายกัน)
    → ทำให้ทุก position มี "ลายนิ้วมือ" เฉพาะตัวที่โมเดลแยกแยะได้
"""

# ================================================================
# (x) ทำไม PE ถึงใช้ทั้ง sin และ cos — ถ้าใช้ sin อย่างเดียวจะเกิดปัญหาอะไร
# W 3.1 : X ทำไม PE ถึงใช้ทั้ง sin และ cos — ถ้าใช้ sin อย่างเดียวจะเกิดปัญหาอะไร
# ================================================================
print("=" * 70)
print("  (x) ทำไมต้องใช้ทั้ง sin และ cos")
print("=" * 70)

# พิสูจน์ปัญหา: sin(pos) ซ้ำค่าเมื่อ pos = π - x และ pos = x
# เช่น sin(0) = sin(π) = sin(2π) = 0
print(f"\n  ▶ ปัญหาของ sin อย่างเดียว:")
print(f"\n    sin(0)     = {np.sin(0):+.4f}")
print(f"    sin(π)     = {np.sin(np.pi):+.4f}   ← เท่ากับ sin(0)")
print(f"    sin(2π)    = {np.sin(2*np.pi):+.4f}  ← เท่ากับ sin(0)")
print(f"    sin(π/2)   = {np.sin(np.pi/2):+.4f}")
print(f"    sin(π-π/2) = {np.sin(np.pi - np.pi/2):+.4f}  ← เท่ากับ sin(π/2)")

print(f"""
  ▶ 3 เหตุผลที่ต้องใช้ทั้ง sin + cos:

    1. แก้ปัญหา ambiguity (ค่าซ้ำ):
       sin(x) เป็น periodic + symmetric → ค่าเดียวกันได้จาก 2 มุม
       เช่น sin(π/4) = sin(3π/4) → 2 position แตกต่างกันได้ vector เท่ากัน
       → model แยกแยะ position ไม่ได้
       cos(x) มี phase ต่างจาก sin 90° → ช่วย disambiguate

    2. รองรับ relative position (linear transformation):
       ทฤษฎี: PE(pos+k) = PE(pos) · M(k)  เมื่อ M(k) เป็น rotation matrix
       เป็นไปได้เพราะ มี sin คู่กับ cos (sin/cos identity)
       → model เรียนรู้ "ห่างกัน k positions" ได้ง่ายผ่าน linear projection

    3. Full circle representation:
       (sin, cos) คือ unit vector บนวงกลม → ทุก position = จุดบนวงกลม
       มี sin อย่างเดียว = แค่ projection บนแกน y → ข้อมูลสูญหาย
       มี cos อย่างเดียว = แค่ projection บนแกน x → เหมือนกัน
       มีทั้งคู่ = (cos, sin) เก็บมุมจริง

  ▶ พิสูจน์ใน PE ของเรา:
""")

"""
======================================================================
  (x) ทำไมต้องใช้ทั้ง sin และ cos
======================================================================

  ▶ ปัญหาของ sin อย่างเดียว:

    sin(0)     = +0.0000
    sin(π)     = +0.0000   ← เท่ากับ sin(0)
    sin(2π)    = -0.0000  ← เท่ากับ sin(0)
    sin(π/2)   = +1.0000
    sin(π-π/2) = +1.0000  ← เท่ากับ sin(π/2)

  ▶ 3 เหตุผลที่ต้องใช้ทั้ง sin + cos:

    1. แก้ปัญหา ambiguity (ค่าซ้ำ):
       sin(x) เป็น periodic + symmetric → ค่าเดียวกันได้จาก 2 มุม
       เช่น sin(π/4) = sin(3π/4) → 2 position แตกต่างกันได้ vector เท่ากัน
       → model แยกแยะ position ไม่ได้
       cos(x) มี phase ต่างจาก sin 90° → ช่วย disambiguate

    2. รองรับ relative position (linear transformation):
       ทฤษฎี: PE(pos+k) = PE(pos) · M(k)  เมื่อ M(k) เป็น rotation matrix
       เป็นไปได้เพราะ มี sin คู่กับ cos (sin/cos identity)
       → model เรียนรู้ "ห่างกัน k positions" ได้ง่ายผ่าน linear projection

    3. Full circle representation:
       (sin, cos) คือ unit vector บนวงกลม → ทุก position = จุดบนวงกลม
       มี sin อย่างเดียว = แค่ projection บนแกน y → ข้อมูลสูญหาย
       มี cos อย่างเดียว = แค่ projection บนแกน x → เหมือนกัน
       มีทั้งคู่ = (cos, sin) เก็บมุมจริง

  ▶ พิสูจน์ใน PE ของเรา:

    ถ้าใช้ sin อย่างเดียว — cosine similarity ระหว่าง position 0..4:
      pos 0..4 vs pos 0..4:
      pos 0: +0.00 +0.00 +0.00 +0.00 +0.00
      pos 1: +0.00 +0.82 +0.97 +0.40 -0.30
      pos 2: +0.00 +0.97 +1.22 +0.67 -0.04
      pos 3: +0.00 +0.40 +0.67 +0.78 +0.80
      pos 4: +0.00 -0.30 -0.04 +0.80 +1.65
    → จะเห็นว่ามีค่าใกล้กันหลายคู่ (potential ambiguity)
"""

# พิสูจน์ว่าถ้าใช้ sin อย่างเดียว pos บางคู่จะได้ vector คล้ายกันมาก
sin_only = pe.pe[:, 0::2]   # dim คู่ทั้งหมด (sin)
sim_matrix_sin = sin_only @ sin_only.T
print(f"    ถ้าใช้ sin อย่างเดียว — cosine similarity ระหว่าง position 0..4:")
print(f"      pos 0..4 vs pos 0..4:")
for i in range(5):
    row = "        " + " ".join(f"{sim_matrix_sin[i,j]:+.2f}" for j in range(5))
    print(f"      pos {i}: {row[8:]}")
print(f"    → จะเห็นว่ามีค่าใกล้กันหลายคู่ (potential ambiguity)")

# ================================================================
# (y) แสดงผล pe.encode(X) โดย X เป็น random embeddings shape (5, 16)
# W 3.1 : Y แสดงผล pe.encode(X) โดยที่ X เป็น random embeddings shape (5, 16) — อธิบายว่า encode() ทำอะไร
# ================================================================
print("\n" + "=" * 70)
print("  (y) pe.encode(X) — X random embeddings shape (5, 16)")
print("=" * 70)

rng = np.random.RandomState(42)
X = rng.randn(5, 16) * 0.1   # random embeddings (5 token, 16 dim)
X_encoded = pe.encode(X)

print(f"\n  ▶ Shape:")
print(f"    X.shape         = {X.shape}    ← random word embedding")
print(f"    pe.pe.shape     = {pe.pe.shape}  ← positional encoding lookup table")
print(f"    encoded.shape   = {X_encoded.shape}    ← เท่ากับ X (ไม่เปลี่ยน shape)")

print(f"\n  ▶ encode() ทำอะไร — source code:")
print(f"      def encode(self, X):")
print(f"          return X + self.pe[:X.shape[0], :]")
print(f"")
print(f"    ตัด pe ให้ตรงกับ seq_len ของ X (5 ตัวแรก) แล้ว 'บวก' element-wise")
print(f"    → output[i, j] = X[i, j] + PE[i, j]")

print(f"\n  ▶ ตรวจ:")
print(f"    X[0, :4]         = {[f'{v:+.4f}' for v in X[0, :4]]}")
print(f"    PE[0, :4]        = {[f'{v:+.4f}' for v in pe.pe[0, :4]]}")
print(f"    X+PE [0, :4]     = {[f'{v:+.4f}' for v in X_encoded[0, :4]]}")
print(f"    ผลรวมตรงไหม?      → {np.allclose(X_encoded, X + pe.pe[:5])}")

print(f"""
  ▶ ทำไมแค่ 'บวก' (ไม่ concat):
    1. ไม่เพิ่ม dimension → keep d_model คงที่ผ่าน layer ทั้งหมด
    2. Embedding หา token identity, PE หา token position
       → บวกกัน = vector ที่บรรจุทั้งสอง info
    3. Transformer layer ถัดไป (attention + linear) เรียนรู้
       'แยก' info ทั้งสองออกเองตามที่ต้องการ
    4. Concat (เช่น 16+16=32 dim) จะเพิ่ม parameter โดยไม่จำเป็น

  ▶ ผลที่ได้คือ "embedding ที่รู้ตำแหน่ง" — นำเข้า attention layer ได้เลย
""")

"""
======================================================================
  (y) pe.encode(X) — X random embeddings shape (5, 16)
======================================================================

  ▶ Shape:
    X.shape         = (5, 16)    ← random word embedding
    pe.pe.shape     = (10, 16)  ← positional encoding lookup table
    encoded.shape   = (5, 16)    ← เท่ากับ X (ไม่เปลี่ยน shape)

  ▶ encode() ทำอะไร — source code:
      def encode(self, X):
          return X + self.pe[:X.shape[0], :]

    ตัด pe ให้ตรงกับ seq_len ของ X (5 ตัวแรก) แล้ว 'บวก' element-wise
    → output[i, j] = X[i, j] + PE[i, j]

  ▶ ตรวจ:
    X[0, :4]         = ['+0.0497', '-0.0138', '+0.0648', '+0.1523']
    PE[0, :4]        = ['+0.0000', '+1.0000', '+0.0000', '+1.0000']
    X+PE [0, :4]     = ['+0.0497', '+0.9862', '+0.0648', '+1.1523']
    ผลรวมตรงไหม?      → True

  ▶ ทำไมแค่ 'บวก' (ไม่ concat):
    1. ไม่เพิ่ม dimension → keep d_model คงที่ผ่าน layer ทั้งหมด
    2. Embedding หา token identity, PE หา token position
       → บวกกัน = vector ที่บรรจุทั้งสอง info
    3. Transformer layer ถัดไป (attention + linear) เรียนรู้
       'แยก' info ทั้งสองออกเองตามที่ต้องการ
    4. Concat (เช่น 16+16=32 dim) จะเพิ่ม parameter โดยไม่จำเป็น

  ▶ ผลที่ได้คือ "embedding ที่รู้ตำแหน่ง" — นำเข้า attention layer ได้เลย
"""
