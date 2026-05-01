import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import json
import importlib.util
import numpy as np
import torch
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

# ===== Patch matplotlib ก่อน import w2 source (รัน plt.show ตอน load) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *_args, **_kwargs: None

# ===== Import balance_legal_data จากไฟล์ต้นฉบับ (มี space ในชื่อ) =====
SRC_PATH = Path(__file__).parent / "w2 Thai IP Legal NLP.py"
spec = importlib.util.spec_from_file_location("w2_src", SRC_PATH)
w2_src = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()):
    try:
        spec.loader.exec_module(w2_src)
    except Exception:
        pass
balance_legal_data = w2_src.balance_legal_data

# ===== Import W1Pipeline + load dataset =====
from w1_thai_legal_nlp import W1Pipeline

DATA_PATH = Path(__file__).parent / "data" / "processed" / "thai_ip_dataset.json"
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
LABEL_NAMES = data["metadata"]["label_names"]
samples = data["samples"]

texts = [s["text"] for s in samples]
labels = [s["label"] for s in samples]

# ===== สร้าง TF-IDF features จาก W1 + รัน SMOTE =====
w1 = W1Pipeline(max_features=50)
X = w1.fit_transform(texts)
y = np.array(labels)

print("\n" + "=" * 70)
print("  Run SMOTE on TF-IDF features จาก W1")
print("=" * 70)
X_res, y_res = balance_legal_data(X, y)
print(f"  X.shape (before): {X.shape}")
print(f"  X_res.shape (after): {X_res.shape}")

# ================================================================
# (n) แสดง distribution ก่อน/หลัง SMOTE — class 2 เพิ่มขึ้นอย่างไร
# W 2.2 : N แสดง distribution ก่อนและหลัง SMOTE — class 2 เพิ่มขึ้นอย่างไร
# ================================================================
print("\n" + "=" * 70)
print("  (n) Distribution ก่อน/หลัง SMOTE")
print("=" * 70)

before = Counter(y.tolist())
after = Counter(y_res.tolist())

LABEL_W = 32    # ความกว้างคอลัมน์ label
NUM_W = 8       # ความกว้างคอลัมน์ตัวเลข

print(f"\n  {'Class':<{LABEL_W}} {'Before':>{NUM_W}} {'After':>{NUM_W}} {'Δ':>{NUM_W}}")
print(f"  {'-' * LABEL_W} {'-' * NUM_W} {'-' * NUM_W} {'-' * NUM_W}")
for cls in sorted(before.keys()):
    b = before[cls]
    a = after[cls]
    diff = a - b
    label = f"Class {cls} ({LABEL_NAMES[cls]})"
    print(f"  {label:<{LABEL_W}} {b:>{NUM_W}} {a:>{NUM_W}} {diff:>+{NUM_W}}")

print(f"\n  ▶ Class 2 (ไม่ละเมิด) เพิ่มขึ้นอย่างไร:")
print(f"     ก่อน: {before[2]} ตัวอย่าง (minority class — {before[2]/sum(before.values())*100:.1f}% ของทั้งหมด)")
print(f"     หลัง: {after[2]} ตัวอย่าง (= ขนาดของ majority class)")
print(f"     SMOTE สังเคราะห์เพิ่ม {after[2] - before[2]} ตัวอย่าง โดย interpolate")
print(f"     จากตัวอย่างจริงและ k-nearest neighbors (ดูคำอธิบายข้อ p)")

"""
======================================================================
  (n) Distribution ก่อน/หลัง SMOTE
======================================================================

  Class                              Before    After        Δ
  -------------------------------- -------- -------- --------
  Class 0 (ละเมิดสิทธิบัตร)              40       40       +0
  Class 1 (ละเมิดลิขสิทธิ์)              20       40      +20
  Class 2 (ไม่ละเมิด)                     6       40      +34

  ▶ Class 2 (ไม่ละเมิด) เพิ่มขึ้นอย่างไร:
     ก่อน: 6 ตัวอย่าง (minority class — 9.1% ของทั้งหมด)
     หลัง: 40 ตัวอย่าง (= ขนาดของ majority class)
     SMOTE สังเคราะห์เพิ่ม 34 ตัวอย่าง โดย interpolate
     จากตัวอย่างจริงและ k-nearest neighbors (ดูคำอธิบายข้อ p)
"""

# ================================================================
# (o) อธิบายเงื่อนไข min_samples > 1 ใน balance_legal_data()
# W 2.2 : O อธิบายเงื่อนไข min_samples > 1 ในโค้ด balance_legal_data() — ถ้าคลาส 2 เหลือ 1 ตัวอย่างจะเกิดอะไรขึ้น
# ================================================================
print("\n" + "=" * 70)
print("  (o) เงื่อนไข min_samples > 1 ใน balance_legal_data()")
print("=" * 70)

print(f"""
  โค้ดต้นฉบับ:
    min_samples = min(counts.values())
    if min_samples > 1:
        sampler = SMOTE(k_neighbors=min(5, min_samples-1), random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)

  เหตุผลของเงื่อนไข:
    SMOTE ต้องการ k-nearest neighbors อย่างน้อย 1 ตัว (k_neighbors >= 1)
    ถ้า class ใด class หนึ่งมีแค่ 1 ตัวอย่าง → ไม่มี neighbor ให้ interpolate
    → SMOTE จะ raise error

  ▶ ทดลองสมมุติ: ถ้า class 2 เหลือ 1 ตัวอย่างจะเกิดอะไรขึ้น
""")

# จำลองสถานการณ์: เก็บ class 2 แค่ 1 ตัวอย่าง
mask = ~((y == 2) & (np.arange(len(y)) > np.where(y == 2)[0][0]))
y_reduced = y[mask]
X_reduced = X[mask]
print(f"     y_reduced distribution: {Counter(y_reduced.tolist())}")
print(f"     min_samples = {min(Counter(y_reduced.tolist()).values())}")
print(f"     → เงื่อนไข min_samples > 1 เป็น False")
print(f"     → fallback ไปใช้ RandomOverSampler แทน SMOTE")
print(f"")
print(f"  ความต่าง:")
print(f"     SMOTE              : สังเคราะห์ตัวอย่างใหม่ที่ 'อยู่ระหว่าง' ตัวจริง")
print(f"                          (ป้องกัน overfitting แต่ต้องการ ≥2 samples/class)")
print(f"     RandomOverSampler  : copy ตัวอย่างเดิมซ้ำ ๆ จนได้ขนาดเท่า majority")
print(f"                          (overfit ง่ายแต่ใช้ได้แม้มี 1 sample/class)")

"""
======================================================================
  (o) เงื่อนไข min_samples > 1 ใน balance_legal_data()
======================================================================

  โค้ดต้นฉบับ:
    min_samples = min(counts.values())
    if min_samples > 1:
        sampler = SMOTE(k_neighbors=min(5, min_samples-1), random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)

  เหตุผลของเงื่อนไข:
    SMOTE ต้องการ k-nearest neighbors อย่างน้อย 1 ตัว (k_neighbors >= 1)
    ถ้า class ใด class หนึ่งมีแค่ 1 ตัวอย่าง → ไม่มี neighbor ให้ interpolate
    → SMOTE จะ raise error

  ▶ ทดลองสมมุติ: ถ้า class 2 เหลือ 1 ตัวอย่างจะเกิดอะไรขึ้น

     y_reduced distribution: Counter({0: 40, 1: 20, 2: 1})
     min_samples = 1
     → เงื่อนไข min_samples > 1 เป็น False
     → fallback ไปใช้ RandomOverSampler แทน SMOTE

  ความต่าง:
     SMOTE              : สังเคราะห์ตัวอย่างใหม่ที่ 'อยู่ระหว่าง' ตัวจริง
                          (ป้องกัน overfitting แต่ต้องการ ≥2 samples/class)
     RandomOverSampler  : copy ตัวอย่างเดิมซ้ำ ๆ จนได้ขนาดเท่า majority
                          (overfit ง่ายแต่ใช้ได้แม้มี 1 sample/class)
"""

# ================================================================
# (p) SMOTE สร้างตัวอย่างใหม่ด้วยวิธี interpolation อธิบายกลไกใน feature space
# W 2.2 : P SMOTE สร้างตัวอย่างใหม่ด้วยวิธี interpolation อธิบายกลไกใน feature space
# ================================================================
print("\n" + "=" * 70)
print("  (p) กลไก SMOTE Interpolation ใน feature space")
print("=" * 70)

print(f"""
  อัลกอริทึม SMOTE สำหรับ minority class:

    1. เลือก minority sample x_i (1 จุดใน feature space {X.shape[1]} มิติ)
    2. หา k nearest neighbors ของ x_i ใน minority class เดียวกัน
       (ใน balance_legal_data() ใช้ k_neighbors = min(5, min_samples-1))
    3. สุ่มเลือก neighbor 1 ตัว = x_nn
    4. สังเคราะห์ตัวอย่างใหม่: x_new = x_i + λ × (x_nn - x_i)
       โดย λ ∈ [0, 1] สุ่มแบบ uniform
    5. ทำซ้ำจนได้จำนวนตัวอย่างเท่ากับ majority class

  ▶ Geometry:
     x_new อยู่บน "เส้นตรง" ระหว่าง x_i กับ x_nn ใน feature space
     ไม่ใช่จุดเดิม → ป้องกัน overfitting จาก duplicate
     ไม่ใช่จุดสุ่มมั่ว → คงคุณสมบัติของ minority class ไว้

  ▶ ตัวอย่างจริงจาก dataset เรา:
     Class 2 มี 6 sample จุดใน 50-D space
     k_neighbors = min(5, 6-1) = 5
     SMOTE สร้างเพิ่มจนครบ {after[2]} sample ทั้งหมดอยู่ใน convex hull ของ class 2 เดิม
""")

"""
======================================================================
  (p) กลไก SMOTE Interpolation ใน feature space
======================================================================

  อัลกอริทึม SMOTE สำหรับ minority class:

    1. เลือก minority sample x_i (1 จุดใน feature space 50 มิติ)
    2. หา k nearest neighbors ของ x_i ใน minority class เดียวกัน
       (ใน balance_legal_data() ใช้ k_neighbors = min(5, min_samples-1))
    3. สุ่มเลือก neighbor 1 ตัว = x_nn
    4. สังเคราะห์ตัวอย่างใหม่: x_new = x_i + λ × (x_nn - x_i)
       โดย λ ∈ [0, 1] สุ่มแบบ uniform
    5. ทำซ้ำจนได้จำนวนตัวอย่างเท่ากับ majority class

  ▶ Geometry:
     x_new อยู่บน "เส้นตรง" ระหว่าง x_i กับ x_nn ใน feature space
     ไม่ใช่จุดเดิม → ป้องกัน overfitting จาก duplicate
     ไม่ใช่จุดสุ่มมั่ว → คงคุณสมบัติของ minority class ไว้

  ▶ ตัวอย่างจริงจาก dataset เรา:
     Class 2 มี 6 sample จุดใน 50-D space
     k_neighbors = min(5, 6-1) = 5
     SMOTE สร้างเพิ่มจนครบ 40 sample ทั้งหมดอยู่ใน convex hull ของ class 2 เดิม
"""

# ================================================================
# (q) แปลง X_res เป็น 3D tensor สำหรับ LSTM แล้วแสดง shape (batch, seq_len, input_size)
# W 2.2 : Q แปลง X_res เป็น 3D tensor สำหรับ LSTM แล้วแสดง shape พร้อมอธิบาย (batch, seq_len, input_size)
# ================================================================
print("\n" + "=" * 70)
print("  (q) แปลง X_res เป็น 3D tensor สำหรับ LSTM")
print("=" * 70)

X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
X_res_3d = X_res_tensor.unsqueeze(1)
y_res_tensor = torch.tensor(y_res, dtype=torch.long)

print(f"""
  Step-by-step:

    1. แปลง numpy → torch tensor (float32 สำหรับ input):
       X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
       shape: {tuple(X_res_tensor.shape)}  ← (n_samples, n_features)

    2. เพิ่มมิติ seq_len ด้วย unsqueeze(1):
       X_res_3d = X_res_tensor.unsqueeze(1)
       shape: {tuple(X_res_3d.shape)}  ← (batch, seq_len, input_size)

    3. แปลง y → long tensor (สำหรับ CrossEntropyLoss):
       y_res_tensor = torch.tensor(y_res, dtype=torch.long)
       shape: {tuple(y_res_tensor.shape)}

  ▶ ความหมายแต่ละมิติของ 3D tensor (batch_first=True):

    batch      = {X_res_3d.shape[0]}   จำนวน document ต่อ batch (ทั้งหมดที่ผ่าน SMOTE)
    seq_len    = {X_res_3d.shape[1]}   ความยาว sequence (=1 เพราะ TF-IDF เป็น vector
                                       เดียวต่อ doc ไม่ใช่ลำดับ token)
    input_size = {X_res_3d.shape[2]}  จำนวน feature ต่อ time step (= max_features)

  ▶ ทำไม seq_len = 1:
     LSTM ออกแบบมาประมวลผล sequence (ลำดับ time step)
     แต่ TF-IDF บีบทั้ง doc เป็น vector เดียว → ไม่มีลำดับ
     ใส่ seq_len=1 = "ทุก doc มี 1 step" → LSTM กลายเป็น MLP เปล่า ๆ
     เหมาะใช้เป็น baseline เท่านั้น — ในงานจริงควรใช้ token embeddings
     ที่มี seq_len = จำนวน token ต่อ doc (เช่น Word2Vec, BERT)
""")

"""
======================================================================
  (q) แปลง X_res เป็น 3D tensor สำหรับ LSTM
======================================================================

  Step-by-step:

    1. แปลง numpy → torch tensor (float32 สำหรับ input):
       X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
       shape: (120, 50)  ← (n_samples, n_features)

    2. เพิ่มมิติ seq_len ด้วย unsqueeze(1):
       X_res_3d = X_res_tensor.unsqueeze(1)
       shape: (120, 1, 50)  ← (batch, seq_len, input_size)

    3. แปลง y → long tensor (สำหรับ CrossEntropyLoss):
       y_res_tensor = torch.tensor(y_res, dtype=torch.long)
       shape: (120,)

  ▶ ความหมายแต่ละมิติของ 3D tensor (batch_first=True):

    batch      = 120   จำนวน document ต่อ batch (ทั้งหมดที่ผ่าน SMOTE)
    seq_len    = 1   ความยาว sequence (=1 เพราะ TF-IDF เป็น vector
                                       เดียวต่อ doc ไม่ใช่ลำดับ token)
    input_size = 50  จำนวน feature ต่อ time step (= max_features)

  ▶ ทำไม seq_len = 1:
     LSTM ออกแบบมาประมวลผล sequence (ลำดับ time step)
     แต่ TF-IDF บีบทั้ง doc เป็น vector เดียว → ไม่มีลำดับ
     ใส่ seq_len=1 = "ทุก doc มี 1 step" → LSTM กลายเป็น MLP เปล่า ๆ
     เหมาะใช้เป็น baseline เท่านั้น — ในงานจริงควรใช้ token embeddings
     ที่มี seq_len = จำนวน token ต่อ doc (เช่น Word2Vec, BERT)
"""
