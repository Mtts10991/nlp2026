import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import json
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ===== Patch matplotlib (source file รัน plt.show ตอน load) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Tahoma'   # font ที่มี Thai glyph (Windows ติดมาด้วย)
plt.show = lambda *_args, **_kwargs: None

# ===== Import จาก w2 source (LegalLSTM, LegalBiLSTM, balance_legal_data) =====
SRC_PATH = Path(__file__).parent / "w2 Thai IP Legal NLP.py"
spec = importlib.util.spec_from_file_location("w2_src", SRC_PATH)
w2_src = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()):
    try:
        spec.loader.exec_module(w2_src)
    except Exception:
        pass

LegalLSTM = w2_src.LegalLSTM
LegalBiLSTM = w2_src.LegalBiLSTM
balance_legal_data = w2_src.balance_legal_data
train_and_evaluate = w2_src.train_and_evaluate

# source's train_and_evaluate เรียก model_class() ไม่มี arg → ต้องใช้ default
# patch default input_dim ให้ตรงกับ TF-IDF dimension (50) ที่ใช้ใน exercise นี้
# ทำที่ runtime ไม่แก้ source file
LegalLSTM.__init__.__defaults__ = (50, 32, 3)    # (input_dim, hidden_dim, output_dim)
LegalBiLSTM.__init__.__defaults__ = (50, 32, 3)

# ===== W1Pipeline + dataset =====
from w1_thai_legal_nlp import W1Pipeline

DATA_PATH = Path(__file__).parent / "data" / "processed" / "thai_ip_dataset.json"
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
LABEL_NAMES = data["metadata"]["label_names"]
samples = data["samples"]

texts = [s["text"] for s in samples]
labels = [s["label"] for s in samples]

# ===== TF-IDF + SMOTE + 3D tensor =====
w1 = W1Pipeline(max_features=50)
X = w1.fit_transform(texts)
y = np.array(labels)
with redirect_stdout(io.StringIO()):
    X_res, y_res = balance_legal_data(X, y)

X_3d = torch.tensor(X_res, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_res, dtype=torch.long)

# ===== Train both models (inline mirror ของ train_and_evaluate ใน source — =====
# ===== exposed y_pred + confusion matrix สำหรับวิเคราะห์ข้อ v) =====
def train_and_predict(model_class, X, y, weights, epochs=50):
    torch.manual_seed(42)
    # ต้องส่ง input_dim = X.shape[-1] เพราะ source class hardcode input_dim=5
    model = model_class(input_dim=X.shape[-1])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X), dim=1).numpy()
    return y_pred

weights = torch.tensor([1.0, 2.0, 2.0])  # Cost-sensitive (hardcoded ใน train_and_evaluate)
y_true = y_tensor.numpy()
class_list = ["ไม่ละเมิด", "ละเมิดสิทธิบัตร", "ละเมิดลิขสิทธิ์"]  # ตามโจทย์
# ⚠️  หมายเหตุ: dataset จริงใช้ index 0=ละเมิดสิทธิบัตร, 1=ละเมิดลิขสิทธิ์, 2=ไม่ละเมิด
#     class_list นี้สลับลำดับ → label ใน report จะถูก map ผิดกับข้อมูลจริง

print("\n" + "=" * 70)
print("  Train LegalLSTM และ LegalBiLSTM บน SMOTE-balanced TF-IDF")
print("=" * 70)
print(f"  Input shape : {tuple(X_3d.shape)} (batch, seq_len, features)")
print(f"  Weights     : {weights.tolist()}  (cost-sensitive — hardcoded ใน source)")
print(f"  class_list  : {class_list}")

# ================================================================
# (r) แสดง classification report ของทั้ง 2 โมเดล — เปรียบเทียบ F1 class 2
# W 2.3 : R แสดง classification report ของทั้ง 2 โมเดล — โมเดลใดมี F1-score ดีกว่าสำหรับ class 2 (minority)
# ================================================================
print("\n" + "=" * 70)
print("  (r) Classification Report — LSTM vs BiLSTM")
print("=" * 70)

# โค้ดบังคับตามโจทย์ — ใช้ LegalLSTM/LegalBiLSTM ตรง ๆ
# (input_dim=50 ถูก patch ที่ runtime ด้านบน เพื่อให้ model_class() ใน source ใช้ได้)
with redirect_stdout(io.StringIO()):
    report_lstm   = train_and_evaluate(LegalLSTM,   "LSTM",   X_3d, y_tensor, class_list)
    report_bilstm = train_and_evaluate(LegalBiLSTM, "BiLSTM", X_3d, y_tensor, class_list)

print(f"\n  ─── LegalLSTM (จาก train_and_evaluate) ───")
print(report_lstm)
print(f"  ─── LegalBiLSTM (จาก train_and_evaluate) ───")
print(report_bilstm)

# train_and_evaluate ของ source return แค่ text report — ต้อง predict ใหม่
# เพื่อเก็บ y_pred + confusion matrix สำหรับข้อ v
y_pred_lstm = train_and_predict(LegalLSTM, X_3d, y_tensor, weights)
y_pred_bilstm = train_and_predict(LegalBiLSTM, X_3d, y_tensor, weights)

# เปรียบเทียบ F1 ของ class 2 ตรง ๆ
rep_lstm = classification_report(y_true, y_pred_lstm, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
rep_bi = classification_report(y_true, y_pred_bilstm, target_names=LABEL_NAMES, output_dict=True, zero_division=0)

print(f"  ▶ F1-score เปรียบเทียบทุก class:")
LABEL_W = 25
print(f"    {'Class':<{LABEL_W}} {'LSTM':>10} {'BiLSTM':>10} {'Δ':>10}")
print(f"    {'-' * LABEL_W} {'-' * 10} {'-' * 10} {'-' * 10}")
for name in LABEL_NAMES:
    f1_l = rep_lstm[name]["f1-score"]
    f1_b = rep_bi[name]["f1-score"]
    diff = f1_b - f1_l
    print(f"    {name:<{LABEL_W}} {f1_l:>10.3f} {f1_b:>10.3f} {diff:>+10.3f}")

"""
======================================================================
  (r) Classification Report — LSTM vs BiLSTM
======================================================================

  ─── LegalLSTM ───
                 precision    recall  f1-score   support

ละเมิดสิทธิบัตร       1.00      0.95      0.97        40
ละเมิดลิขสิทธิ์       0.75      0.45      0.56        40
      ไม่ละเมิด       0.59      0.85      0.69        40

       accuracy                           0.75       120
      macro avg       0.78      0.75      0.74       120
   weighted avg       0.78      0.75      0.74       120

  ─── LegalBiLSTM ───
                 precision    recall  f1-score   support

ละเมิดสิทธิบัตร       1.00      0.95      0.97        40
ละเมิดลิขสิทธิ์       0.75      0.45      0.56        40
      ไม่ละเมิด       0.59      0.85      0.69        40

       accuracy                           0.75       120
      macro avg       0.78      0.75      0.74       120
   weighted avg       0.78      0.75      0.74       120

  ▶ F1-score เปรียบเทียบทุก class:
    Class                           LSTM     BiLSTM          Δ
    ------------------------- ---------- ---------- ----------
    ละเมิดสิทธิบัตร                0.974      0.974     +0.000
    ละเมิดลิขสิทธิ์                0.562      0.562     +0.000
    ไม่ละเมิด                      0.694      0.694     +0.000
"""

# ================================================================
# (s) อธิบายทำไม BiLSTM ดีกว่า LSTM สำหรับ legal text — long-range dependency
# W 2.3 : S อธิบายทำไม BiLSTM จึงมักดีกว่า LSTM สำหรับ legal text — เชื่อมกับแนวคิด long-range dependency
# ================================================================
print("\n" + "=" * 70)
print("  (s) ทำไม BiLSTM ดีกว่า LSTM สำหรับ legal text")
print("=" * 70)

print(f"""
  LSTM (Unidirectional):
    อ่าน sequence จากซ้ายไปขวาเท่านั้น → hidden state ที่ time t รู้แค่
    context ก่อนหน้า ไม่รู้ว่าหลังจากนี้จะมีอะไร

  BiLSTM (Bidirectional):
    มี 2 LSTM วิ่งสวนทางกัน — ตัวหนึ่งซ้าย→ขวา อีกตัวขวา→ซ้าย
    แล้ว concat hidden state ทั้งสอง → ทุก time step เห็นทั้ง past + future

  ▶ ทำไมเหมาะกับ legal text:

    1. Long-range dependency ของกฎหมายมักเป็นแบบ "หน้า-หลัง"
       เช่น "จำเลย ___ ตามมาตรา 77 พ.ร.บ.สิทธิบัตร"
       - LSTM ตอนเจอ "จำเลย" ยังไม่รู้ว่าเป็น context สิทธิบัตร
       - BiLSTM เห็น "พ.ร.บ.สิทธิบัตร" จากทิศทาง→ → ใส่ context ให้ "จำเลย"

    2. Cataphora (สรรพนามล่วงหน้า) ในไทย:
       เช่น "เขาผลิตยาที่เลียนแบบสิทธิบัตรของบริษัท X"
       - "เขา" = บริษัทไหน? LSTM ต้องรอจนจบประโยคถึงรู้
       - BiLSTM ใส่ context "บริษัท X" ย้อนกลับมาที่ "เขา" ได้ทันที

    3. Negation scope: "ไม่ได้ละเมิดสิทธิบัตร"
       - LSTM อาจ classify เป็น 'ละเมิด' ถ้าน้ำหนักของ "ละเมิด" ดึงไปก่อน
       - BiLSTM เห็น "ไม่ได้" จากทิศทาง→ → ปรับ representation ของ "ละเมิด"

  ⚠️  ข้อแม้ใน exercise นี้: seq_len=1 → BiLSTM ก็ไม่ได้ประโยชน์เลย
       ผลลัพธ์ที่เห็นจึงไม่สะท้อน advantage ของ BiLSTM อย่างแท้จริง
       ต้องใช้ token embedding (seq_len = N tokens) ถึงจะเห็นความต่างชัด
""")

# ================================================================
# (t) Cost-sensitive weights [1.0, 2.0, 2.0] — False Negative class 1 มีผลอย่างไร
# W 2.3 : T Cost-sensitive weights [1.0, 2.0, 2.0] หมายความว่าอย่างไร — False Negative ของ class 1 มีผลกระทบต่อคดีอย่างไร
# ================================================================
print("\n" + "=" * 70)
print("  (t) Cost-sensitive weights [1.0, 2.0, 2.0]")
print("=" * 70)

print(f"""
  ความหมาย: weights ใน CrossEntropyLoss(weight=weights) คูณ loss ตาม class
    weights[0] = 1.0  → class 0 ({LABEL_NAMES[0]})
    weights[1] = 2.0  → class 1 ({LABEL_NAMES[1]})  ← penalty 2 เท่า
    weights[2] = 2.0  → class 2 ({LABEL_NAMES[2]})  ← penalty 2 เท่า

  ตีความเชิงคดี:
    Loss สูง 2 เท่าสำหรับ class 1 และ 2 = "model ที่ผิดในสองคลาสนี้
    จะถูกลงโทษหนักกว่า" → optimizer จะพยายามไม่ผิดที่ class 1, 2

  ▶ ผลของ False Negative class 1 ({LABEL_NAMES[1]}) ในคดีจริง:

    FN = "เป็น {LABEL_NAMES[1]} จริง แต่ model ทาย 'ไม่ใช่'"

    ผลกระทบ:
    1. คดีลิขสิทธิ์จริงไม่ถูกตรวจจับ → ผู้เสียหายไม่ได้รับความคุ้มครอง
    2. หากเป็นระบบ pre-screening ของศาล → คดีตกหล่นจาก pipeline
    3. เจ้าของลิขสิทธิ์ (เช่น นักเขียน, ค่ายเพลง) เสียรายได้ต่อเนื่อง
    4. ผู้กระทำผิดได้รับสัญญาณว่า "ทำได้ ไม่โดนจับ" → recidivism

    เพราะ FN class 1 คือ "พลาดคดีที่มีอยู่จริง" ค่าใช้จ่ายทางสังคม
    สูงกว่า FP (กล่าวหาผิด) — จึงต้องให้ weight สูงให้ model ไวต่อ class 1
""")

"""


======================================================================
  (t) Cost-sensitive weights [1.0, 2.0, 2.0]
======================================================================

  ความหมาย: weights ใน CrossEntropyLoss(weight=weights) คูณ loss ตาม class
    weights[0] = 1.0  → class 0 (ละเมิดสิทธิบัตร)
    weights[1] = 2.0  → class 1 (ละเมิดลิขสิทธิ์)  ← penalty 2 เท่า
    weights[2] = 2.0  → class 2 (ไม่ละเมิด)  ← penalty 2 เท่า

  ตีความเชิงคดี:
    Loss สูง 2 เท่าสำหรับ class 1 และ 2 = "model ที่ผิดในสองคลาสนี้
    จะถูกลงโทษหนักกว่า" → optimizer จะพยายามไม่ผิดที่ class 1, 2

  ▶ ผลของ False Negative class 1 (ละเมิดลิขสิทธิ์) ในคดีจริง:

    FN = "เป็น ละเมิดลิขสิทธิ์ จริง แต่ model ทาย 'ไม่ใช่'"

    ผลกระทบ:
    1. คดีลิขสิทธิ์จริงไม่ถูกตรวจจับ → ผู้เสียหายไม่ได้รับความคุ้มครอง
    2. หากเป็นระบบ pre-screening ของศาล → คดีตกหล่นจาก pipeline
    3. เจ้าของลิขสิทธิ์ (เช่น นักเขียน, ค่ายเพลง) เสียรายได้ต่อเนื่อง
    4. ผู้กระทำผิดได้รับสัญญาณว่า "ทำได้ ไม่โดนจับ" → recidivism

    เพราะ FN class 1 คือ "พลาดคดีที่มีอยู่จริง" ค่าใช้จ่ายทางสังคม
    สูงกว่า FP (กล่าวหาผิด) — จึงต้องให้ weight สูงให้ model ไวต่อ class 1
"""

# ================================================================
# (u) Xavier Initialization ช่วยอย่างไร — เปรียบเทียบกับ random init
# W 2.3 : U Xavier Initialization ใน nn.init.xavier_uniform_(self.fc.weight) ช่วยอย่างไร — เปรียบเทียบกับ random init
# ================================================================
print("=" * 70)
print("  (u) Xavier Initialization vs Random Init")
print("=" * 70)

print(f"""
  Xavier (Glorot) Uniform Init:
    weight ~ Uniform[-a, +a]  โดย  a = sqrt(6 / (fan_in + fan_out))
    fan_in  = จำนวน input neurons
    fan_out = จำนวน output neurons

  เปรียบเทียบ initial variance:
    Random uniform [-1, 1]   → variance ≈ 0.33  (ใหญ่เกินไป)
    Random normal N(0, 1)    → variance = 1.00  (ใหญ่มาก)
    Xavier (fan_in=64)       → variance ≈ 0.03  (พอดี)

  ▶ ปัญหาของ random init ที่ Xavier แก้:

    1. Vanishing gradient: weight เริ่มต้นเล็กมาก → output ทุก layer
       เข้าใกล้ 0 → gradient ที่ backprop กลับมาก็เล็ก → train ไม่เดิน

    2. Exploding gradient: weight เริ่มต้นใหญ่มาก → output โต
       ทุก layer → activation saturate (tanh, sigmoid) → gradient = 0
       (ปัญหาเดียวกันคนละ symptom)

    3. Slow convergence: ต้องใช้ learning rate ต่ำ + epochs เยอะ
       เพื่อให้ weights ปรับมาอยู่ใน range ที่เหมาะ

  ▶ Xavier formula ออกแบบให้:
    Var(output) = Var(input)  ← signal magnitude คงตัวข้าม layer
    Var(grad_input) = Var(grad_output)  ← gradient magnitude คงตัว backprop

    ผลคือ network train ได้เร็วขึ้น เสถียรขึ้น โดยไม่ต้อง batch norm

  ▶ ใน LegalBiLSTM/LegalLSTM:
    nn.init.xavier_uniform_(self.fc.weight) ปรับ output layer (fc)
    ส่วน LSTM layer ใช้ default init ของ PyTorch (orthogonal สำหรับ
    recurrent weight, zero สำหรับ bias) ซึ่งก็เหมาะกับ RNN อยู่แล้ว
""")

# ================================================================
# (v) Confusion Matrix — class คู่ไหนสับสนกันมากที่สุด แล้วแก้ยังไง
# W 2.3 : V จาก Confusion Matrix ระบุว่าโมเดลสับสนระหว่าง class ใดมากที่สุด — เสนอวิธีแก้ไข
# ================================================================
print("=" * 70)
print("  (v) Confusion Matrix Analysis")
print("=" * 70)

cm_lstm = confusion_matrix(y_true, y_pred_lstm)
cm_bilstm = confusion_matrix(y_true, y_pred_bilstm)

def print_cm(title, cm):
    print(f"\n  ─── {title} ───")
    print(f"    {'Actual\\Pred':<22} {LABEL_NAMES[0]:<18} {LABEL_NAMES[1]:<18} {LABEL_NAMES[2]:<18}")
    for i, row in enumerate(cm):
        print(f"    {LABEL_NAMES[i]:<22} {row[0]:<18} {row[1]:<18} {row[2]:<18}")

print_cm("LegalLSTM", cm_lstm)
print_cm("LegalBiLSTM", cm_bilstm)

# วิเคราะห์ pair ที่สับสนมากที่สุด (ไม่นับ diagonal)
def worst_confusion(cm):
    n = cm.shape[0]
    worst_pair = None
    worst_count = -1
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > worst_count:
                worst_count = cm[i, j]
                worst_pair = (i, j)
    return worst_pair, worst_count

(i_l, j_l), n_l = worst_confusion(cm_lstm)
(i_b, j_b), n_b = worst_confusion(cm_bilstm)

print(f"""
  ▶ คู่ class ที่สับสนมากที่สุด:

    LegalLSTM   : {LABEL_NAMES[i_l]} → ทาย {LABEL_NAMES[j_l]} ({n_l} ครั้ง)
    LegalBiLSTM : {LABEL_NAMES[i_b]} → ทาย {LABEL_NAMES[j_b]} ({n_b} ครั้ง)

  ▶ วิธีแก้ไขความสับสน:

    1. เพิ่ม training data ที่แยกความต่างระหว่าง 2 class นี้ชัด ๆ
       (เช่น เพิ่มประโยคที่มี keyword เฉพาะของแต่ละ class)

    2. Feature engineering:
       - เพิ่ม IP_TYPE ที่ extract จาก W1 เป็น categorical feature
       - เพิ่ม STATUTE reference (ถ้า class แต่ละตัวอ้างมาตราต่างกัน)
       - ใช้ Legal Hierarchy weight จาก W1 (Physics Gate Weight)

    3. ใช้ feature ที่มี contextual meaning:
       - แทน TF-IDF ด้วย Word2Vec / WangchanBERTa embedding
       - seq_len > 1 ให้ LSTM ใช้ลำดับ token ได้จริง

    4. ปรับ class weights ที่ละเอียดขึ้น:
       - ถ้า class A สับสนกับ B บ่อย → เพิ่ม weight ของ B ขึ้นไปอีก
       - หรือใช้ Focal Loss แทน CrossEntropyLoss (down-weight easy examples)

    5. Ensemble: รวมผล LSTM + BiLSTM + rule-based (W1 entity extractor)
       voting → ความสับสนของแต่ละ model หักล้างกัน

  ⚠️  หมายเหตุ: เพราะ train + eval บน data ชุดเดียวกัน → numbers สูงเกินจริง
       ในการประเมินจริงต้อง split train/val/test แยกกัน
""")

"""
======================================================================
  (v) Confusion Matrix Analysis
======================================================================

  ─── LegalLSTM ───
    Actual\Pred            ละเมิดสิทธิบัตร    ละเมิดลิขสิทธิ์    ไม่ละเมิด         
    ละเมิดสิทธิบัตร        38                 0                  2                 
    ละเมิดลิขสิทธิ์        0                  18                 22                
    ไม่ละเมิด              0                  6                  34                

  ─── LegalBiLSTM ───
    Actual\Pred            ละเมิดสิทธิบัตร    ละเมิดลิขสิทธิ์    ไม่ละเมิด         
    ละเมิดสิทธิบัตร        38                 0                  2                 
    ละเมิดลิขสิทธิ์        0                  18                 22                
    ไม่ละเมิด              0                  6                  34                

  ▶ คู่ class ที่สับสนมากที่สุด:

    LegalLSTM   : ละเมิดลิขสิทธิ์ → ทาย ไม่ละเมิด (22 ครั้ง)
    LegalBiLSTM : ละเมิดลิขสิทธิ์ → ทาย ไม่ละเมิด (22 ครั้ง)

  ▶ วิธีแก้ไขความสับสน:

    1. เพิ่ม training data ที่แยกความต่างระหว่าง 2 class นี้ชัด ๆ
       (เช่น เพิ่มประโยคที่มี keyword เฉพาะของแต่ละ class)

    2. Feature engineering:
       - เพิ่ม IP_TYPE ที่ extract จาก W1 เป็น categorical feature
       - เพิ่ม STATUTE reference (ถ้า class แต่ละตัวอ้างมาตราต่างกัน)
       - ใช้ Legal Hierarchy weight จาก W1 (Physics Gate Weight)

    3. ใช้ feature ที่มี contextual meaning:
       - แทน TF-IDF ด้วย Word2Vec / WangchanBERTa embedding
       - seq_len > 1 ให้ LSTM ใช้ลำดับ token ได้จริง

    4. ปรับ class weights ที่ละเอียดขึ้น:
       - ถ้า class A สับสนกับ B บ่อย → เพิ่ม weight ของ B ขึ้นไปอีก
       - หรือใช้ Focal Loss แทน CrossEntropyLoss (down-weight easy examples)

    5. Ensemble: รวมผล LSTM + BiLSTM + rule-based (W1 entity extractor)
       voting → ความสับสนของแต่ละ model หักล้างกัน

  ⚠️  หมายเหตุ: เพราะ train + eval บน data ชุดเดียวกัน → numbers สูงเกินจริง
       ในการประเมินจริงต้อง split train/val/test แยกกัน
"""
