import sys
sys.stdout.reconfigure(encoding='utf-8')  # กัน UnicodeEncodeError บน Windows console (cp1252)

import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

CLASS_NAMES = ['NO-INFRINGEMENT', 'PATENT', 'COPYRIGHT']
# CLASS_NAMES = ['NO-INFRINGEMENT', 'INFRINGEMENT', 'COUNTERFEIT', 'PATENT', 'TRADEMARK', 'COPYRIGHT']

LEGAL_SYNONYMS = {
    "ละเมิด": ["ละเมิด", "ฝ่าฝืน", "ทำผิด"],
    "ปลอมแปลง": ["ปลอมแปลง", "ปลอม", "เลียนแบบ", "เลียนแบบ", "ทำซ้ำ", "ดัดแปลง"],
    "จำหน่าย": ["จำหน่าย", "ขาย", "จัดจำหน่าย"],
}

def aument_legal_text(text):
    WORD = text.split()
    NEW_WORD = WORD.copy()

    for word, synonyms in LEGAL_SYNONYMS.items():
        if word in text:
            synonym = random.choice(synonyms)
            for i, w in enumerate(NEW_WORD):
                if w == word:
                    NEW_WORD[i] = synonym
    return " ".join(NEW_WORD)
ORIGINAL_TEXT = "จำเลย ละเมิด และ จำหน่าย สินค้า"
AUGMENTED_TEXT = aument_legal_text(ORIGINAL_TEXT)
print(""*50)
print(f"Original Text: {ORIGINAL_TEXT}")
print(f"Augmented Text: {AUGMENTED_TEXT}")
print(""*50)

#2 SMOTE With Fallback
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification
from collections import Counter

def balance_legal_data(INPUT, OUTPUT):
    try:
        COUNTS = Counter(OUTPUT)
        print(f"Original Class Distribution: {COUNTS}")
        # Class น้อยสุดมีกี่ตัวอย่าง
        MIN_SAMPLES = min(COUNTS.values())
        if MIN_SAMPLES > 1:
            SAMPLER = SMOTE(k_neighbors=min(5, MIN_SAMPLES - 1), random_state=7)
        else:
            print("Not enough samples for SMOTE, using RandomOverSampler instead.")
            SAMPLER = RandomOverSampler(random_state=7)
        INPUT_RES, OUTPUT_RES = SAMPLER.fit_resample(INPUT, OUTPUT)
        print(f"Balanced Class Distribution: {Counter(OUTPUT_RES)}")
        return INPUT_RES, OUTPUT_RES
    except Exception as e:
        print(f"Error in counting classes: {e}")

# สร้าง dataset ตัวอย่าง (Class 0 = 100 ตัวอย่าง, Class 1 = 2 ตัวอย่าง)
# INPUT, OUTPUT = make_classification(n_classes=2, class_sep=2, 
#                                     weights=[0.98, 0.02], n_informative=3, 
#                                     n_redundant=1, flip_y=0, n_features=20, 
#                                     n_clusters_per_class=1, n_samples=102, 
#                                     random_state=7)
# INPUT_RES, OUTPUT_RES = balance_legal_data(INPUT, OUTPUT)

INPUT = np.random.randn(60, 5)  # 60 ตัวอย่าง, 5 features
# 6 คลาสไม่สมดุล (ตรงกับ CLASS_NAMES) เพื่อให้ SMOTE มี class ให้ balance
OUTPUT = np.array([0]*25 + [1]*20 + [2]*15)
# OUTPUT = np.array([0]*20 + [1]*15 + [2]*10 + [3]*8 + [4]*5 + [5]*2)
INPUT_RES, OUTPUT_RES = balance_legal_data(INPUT, OUTPUT) # ทดสอบรัน

#3 BiLSTM-CRF (จำลองการใช้ BiLSTM-CRF สำหรับการทำ Named Entity Recognition ในเอกสารกฎหมาย)
import torch
import torch.nn as neural_network

class LegalBiLSTM(neural_network.Module):
    def __init__(self, input_dimension = 16, hidden_dimension = 32, output_dimension = 6):
        super(LegalBiLSTM, self).__init__()
        self.lstm = neural_network.LSTM(input_dimension, hidden_dimension, bidirectional=True, batch_first=True)
        self.fc = neural_network.Linear(hidden_dimension * 2, output_dimension)  # *2 เพราะเป็น BiLSTM
        neural_network.init.xavier_uniform_(self.fc.weight)  # การเริ่มต้นน้ำหนักแบบ Xavier

    def forward(self, OUTPUT):
        LSTM_OUTPUT, _ = self.lstm(OUTPUT)
        # Mean Pooling เพราะเพื่อให้ได้ representation ของทั้ง sequence
        MEAN_POOL = torch.mean(LSTM_OUTPUT, dim=1)
        return self.fc(MEAN_POOL)
    
MODEL = LegalBiLSTM()
sample_input = torch.randn(1, 10, 16)  # Batch size = 1, Sequence length = 10, Input dimension = 16
OUTPUT = MODEL(sample_input)
print(""*50)
print(f"Model Output BiLSTM Shape: {OUTPUT.shape}")
print(f"Model Logical Output: {OUTPUT}")
print(""*50)

# LSTM ทิศทางเดียว (จำลองการใช้ LSTM ทิศทางเดียวสำหรับการทำ Text Classification ในเอกสารกฎหมาย)
class LegalLSTM(neural_network.Module):
    def __init__(self, input_dimension = 16, hidden_dimension = 32, output_dimension = 6):
        super(LegalLSTM, self).__init__()
        self.lstm = neural_network.LSTM(input_dimension, hidden_dimension, bidirectional=False, batch_first=True) # bidirectional=False สำหรับ LSTM ทิศทางเดียว ถ้าเป็น True จะเป็น BiLSTM 2 ทิศทาง
        self.fc = neural_network.Linear(hidden_dimension, output_dimension)
        # การเริ่มต้นน้ำหนักแบบ Xavier เลือกค่า weight ของ fully connected layer ด้วยการเริ่มต้นแบบ Xavier เพื่อช่วยให้การเรียนรู้มีประสิทธิภาพมากขึ้น ในการฝึกโมเดล LSTM ทิศทางเดียวสำหรับการทำ Text Classification ในเอกสารกฎหมาย รอบแรกจะช่วยให้โมเดลเริ่มต้นด้วยน้ำหนักที่เหมาะสม ซึ่งสามารถช่วยให้การเรียนรู้มีประสิทธิภาพมากขึ้นและลดปัญหา vanishing gradient ที่อาจเกิดขึ้นใน LSTM ได้
        neural_network.init.xavier_uniform_(self.fc.weight) 

    def forward(self, OUTPUT):
        LSTM_OUTPUT, _ = self.lstm(OUTPUT)
        # Mean Pooling เพื่อให้ได้ representation ของทั้ง sequence
        MEAN_POOL = torch.mean(LSTM_OUTPUT, dim=1)
        return self.fc(MEAN_POOL)
    
# เทรนโมเดล LSTM ทิศทางเดียว VS BiLSTM (จำลองการเปรียบเทียบผลลัพธ์ของ LSTM ทิศทางเดียวและ BiLSTM ในการทำ Text Classification ในเอกสารกฎหมาย)
def trainModelEvaluate(MODEL, NAME, INPUT, OUTPUT, ClassName):
    # Cost-Sensitive Weight (จำลองการใช้ Cost-Sensitive Weighting ในการฝึกโมเดลเพื่อจัดการกับปัญหาความไม่สมดุลของคลาสในเอกสารกฎหมาย) = FN (False Negatives) จะมีค่าเสียหายสูงกว่า FP (False Positives) ในการจำแนกประเภทเอกสารกฎหมาย เช่น การจำแนกว่าข้อความเป็น "ละเมิด" หรือไม่ ถ้าโมเดลพลาดการจำแนกข้อความที่เป็น "ละเมิด" (False Negative) อาจส่งผลให้เกิดความเสียหายทางกฎหมายหรือการสูญเสียทางการเงิน ในขณะที่การจำแนกข้อความที่ไม่ใช่ "ละเมิด" เป็น "ละเมิด" (False Positive) อาจทำให้เกิดความไม่สะดวกหรือความผิดพลาดในการจัดการเอกสาร แต่จะไม่ส่งผลเสียหายเท่ากับ False Negatives ดังนั้น การใช้ Cost-Sensitive Weighting เพื่อเพิ่มน้ำหนักให้กับ False Negatives จะช่วยให้โมเดลมีความระมัดระวังมากขึ้นในการจำแนกข้อความที่เป็น "ละเมิด" และลดโอกาสในการพลาดการจำแนกข้อความที่สำคัญในเอกสารกฎหมาย

    # ให้ Class 0 (NO-INFRINGEMENT) น้ำหนักต่ำสุด, ที่เหลือให้น้ำหนักสูงเพื่อลด False Negative
    WEIGHTS = torch.tensor([1.0, 2.0, 2.0])
    # WEIGHTS = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    CRITERION = neural_network.CrossEntropyLoss(weight=WEIGHTS) # ใช้ CrossEntropyLoss พร้อมกับ Cost-Sensitive Weighting เพื่อช่วยให้โมเดลมีความระมัดระวังมากขึ้นในการจำแนกข้อความที่เป็น "ละเมิด" และลดโอกาสในการพลาดการจำแนกข้อความที่สำคัญในเอกสารกฎหมาย

    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.001) # ใช้ Adam Optimizer สำหรับการฝึกโมเดล LSTM ทิศทางเดียวและ BiLSTM ในการทำ Text Classification ในเอกสารกฎหมาย เพราะ Adam เป็น Optimizer ที่มีประสิทธิภาพสูงในการปรับแต่งน้ำหนักของโมเดลในระหว่างการฝึก และสามารถช่วยให้โมเดลเรียนรู้ได้เร็วขึ้นและมีประสิทธิภาพมากขึ้นในการจัดการกับปัญหาความไม่สมดุลของคลาสในเอกสารกฎหมาย

    # Simple Training Loop (จำลองการฝึกโมเดลด้วยการทำซ้ำหลายรอบและอัปเดตน้ำหนักของโมเดลในแต่ละรอบ)
    # Epoch คือรอบการฝึกที่โมเดลจะเห็นข้อมูลทั้งหมดหนึ่งครั้ง ในแต่ละ epoch โมเดลจะทำการทำนายผลลัพธ์จากข้อมูลอินพุตและคำนวณค่าความสูญเสีย (loss) โดยใช้ฟังก์ชัน loss ที่กำหนดไว้ จากนั้นจะทำการย้อนกลับ (backpropagation) เพื่อคำนวณกราดิเอนต์ของน้ำหนักในโมเดล และสุดท้ายจะอัปเดตน้ำหนักของโมเดลโดยใช้ optimizer ที่กำหนดไว้ การทำซ้ำหลายรอบนี้ช่วยให้โมเดลเรียนรู้จากข้อมูลและปรับปรุงประสิทธิภาพในการจำแนกประเภทเอกสารกฎหมายได้ดีขึ้น
    for epoch in range(51):  # สมมติว่าเทรน 51 รอบ
        MODEL.train() # กำหนดโมเดลให้ในโหมดการฝึก
        OPTIMIZER.zero_grad() # เซ็ตกราดิเอนต์ของน้ำหนักในโมเดลก่อนการคำนวณกราดิเอนต์ใหม่ในแต่ละรอบการฝึก
        OUTPUT_PRED = MODEL(INPUT) # ทำนายผลลัพธ์จากข้อมูลอินพุตโดยใช้โมเดลที่กำหนดไว้
        LOSS = CRITERION(OUTPUT_PRED, OUTPUT) # คำนวณค่าความสูญเสีย (loss) โดยใช้ฟังก์ชัน loss ที่กำหนดไว้ ซึ่งจะเปรียบเทียบผลลัพธ์ที่โมเดลทำนายกับผลลัพธ์จริง (ground truth) และคำนวณค่าความสูญเสียที่แสดงถึงความแตกต่างระหว่างผลลัพธ์ที่ทำนายและผลลัพธ์จริง
        LOSS.backward() # ทำการย้อนกลับเพื่อคำนวนค่าน้ำหนักของโมเดลโดยใช้กราดิเอนต์ที่คำนวณได้จากการคำนวณค่าความสูญเสีย (loss) ในขั้นตอนก่อนหน้า
        OPTIMIZER.step() # อัปเดตน้ำหนักของโมเดลโดยใช้ optimizer ที่กำหนดไว้ ซึ่งจะปรับแต่งน้ำหนักของโมเดลตามกราดิเอนต์ที่คำนวณได้จากการย้อนกลับในขั้นตอนก่อนหน้า เพื่อให้โมเดลเรียนรู้และปรับปรุงประสิทธิภาพในการจำแนกประเภทเอกสารกฎหมายได้ดีขึ้นในแต่ละรอบการฝึก

    # Evaluate With Confusion Matrix (จำลองการประเมินผลโมเดลด้วยการสร้าง Confusion Matrix เพื่อวิเคราะห์ผลลัพธ์ของการจำแนกประเภทเอกสารกฎหมาย)
    MODEL.eval() # กำหนดโมเดลให้ในโหมดการประเมินผล
    with torch.no_grad(): # ปิดการคำนวณกราดิเอนต์ในระหว่างการประเมินผลเพื่อประหยัดหน่วยความจำและเพิ่มประสิทธิภาพในการประเมินผลโมเดล
        OUTPUT_PRED = MODEL(INPUT) # Predict ทำนายผลลัพธ์จากข้อมูลอินพุตโดยใช้โมเดลที่กำหนดไว้ในโหมดการประเมินผล
        PRED_LABELS = torch.argmax(OUTPUT_PRED, dim=1) # แปลงผลลัพธ์ที่ทำนายเป็นป้ายกำกับ (labels) โดยใช้ฟังก์ชัน argmax เพื่อเลือกคลาสที่มีค่าความน่าจะเป็นสูงสุดสำหรับแต่ละตัวอย่างในข้อมูลอินพุต
        PRED_LABELS = PRED_LABELS.cpu().numpy() # แปลงป้ายกำกับที่ทำนายเป็น NumPy array เพื่อใช้ในการสร้าง Confusion Matrix ในขั้นตอนต่อไป
    # บังคับให้ CM เป็น NxN ของ ClassName ทุกครั้ง แม้บาง class ไม่ปรากฏในผล predict
    CM = confusion_matrix(OUTPUT.cpu().numpy(), PRED_LABELS, labels=list(range(len(ClassName))))
    print(f"Confusion Matrix for {NAME}:\n{CM}\n")

    # Show Heatmap of Confusion Matrix (จำลองการแสดงผล Confusion Matrix ในรูปแบบ Heatmap เพื่อให้เห็นภาพรวมของผลลัพธ์การจำแนกประเภทเอกสารกฎหมายได้ชัดเจนขึ้น)
    plt.figure(figsize=(13, 4))
    sns.heatmap(CM, annot=True, fmt='d', xticklabels=ClassName, yticklabels=ClassName) # แต่ละพารามิเตอร์ในฟังก์ชัน heatmap มีความหมายดังนี้:
    # CM: เป็น Confusion Matrix ที่เราต้องการแสดงผลในรูปแบบ Heatmap
    # annot=True: เป็นการแสดงตัวเลขในแต่ละช่องของ Heatmap ซึ่งจะแสดงจำนวนตัวอย่างที่ถูกจำแนกถูกต้องและผิดพลาดสำหรับแต่ละคลาสในเอกสารกฎหมาย
    # fmt='d': เป็นการกำหนดรูปแบบของตัวเลขที่แสดงใน Heatmap ให้เป็นจำนวนเต็ม (integer) ซึ่งจะช่วยให้ตัวเลขดูชัดเจนและง่ายต่อการอ่านในกรณีที่จำนวนตัวอย่างมีค่ามาก
    plt.title(f'Confusion Matrix for {NAME}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ทดสอบรันเปรียบเทียบผลลัพธ์ของ LSTM ทิศทางเดียวและ BiLSTM ในการทำ Text Classification ในเอกสารกฎหมาย
# แปลง INPUT_RES (numpy 2D) เป็น tensor 3D ให้ตรงกับที่ LSTM ต้องการ: (batch, seq_len, features)
INPUT_TENSOR = torch.tensor(INPUT_RES, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, 5) เหตุผลเพราะ LSTM ต้องการ input เป็น 3D (batch_size, seq_len, input_size) และเราใช้ seq_len=1 เพราะข้อมูลของเรามีแค่ 1 timestep ต่อ sample
OUTPUT_TENSOR = torch.tensor(OUTPUT_RES, dtype=torch.long) # shape: (N,) เพราะ CrossEntropyLoss ต้องการ target เป็น 1D tensor ที่มีค่าเป็น class index สำหรับแต่ละตัวอย่าง
NUM_FEATURES = INPUT_TENSOR.shape[2] # จำนวน features ที่ LSTM จะรับเข้าไปในแต่ละ timestep ซึ่งในที่นี้คือ 5 เพราะเราสร้าง INPUT เป็น (N, 1, 5) และเราจะใช้ NUM_FEATURES นี้ในการกำหนด input_dimension ของโมเดล LSTM และ BiLSTM เพื่อให้ตรงกับจำนวน features ของข้อมูลจริงที่เรามีอยู่

# สร้าง instance ใหม่ของแต่ละโมเดลโดยให้ input_dimension ตรงกับ feature ของข้อมูลจริง
LSTM_MODEL = LegalLSTM(input_dimension=NUM_FEATURES, output_dimension=len(CLASS_NAMES))
BILSTM_MODEL = LegalBiLSTM(input_dimension=NUM_FEATURES, output_dimension=len(CLASS_NAMES))

print(f"Input Shape for LSTM: {INPUT_RES.shape}, Output Shape: {OUTPUT_RES.shape}")

trainModelEvaluate(LSTM_MODEL, "Unidirectional LSTM", INPUT_TENSOR, OUTPUT_TENSOR, CLASS_NAMES)
# trainModelEvaluate(BILSTM_MODEL, "Bidirectional LSTM", INPUT_TENSOR, OUTPUT_TENSOR, CLASS_NAMES)