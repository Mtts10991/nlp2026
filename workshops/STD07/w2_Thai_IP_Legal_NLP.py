import random

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

INPUT = np.random.randn(12,5)  # 12 ตัวอย่าง, 5 features
OUTPUT = np.array([0]*10 + [1]*2)  # Class 0 = 10 ตัวอย่าง, Class 1 = 2 ตัวอย่าง
INPUT_RES, OUTPUT_RES = balance_legal_data(INPUT, OUTPUT) # ทดสอบรัน

#3 BiLSTM-CRF (จำลองการใช้ BiLSTM-CRF สำหรับการทำ Named Entity Recognition ในเอกสารกฎหมาย)
import torch
import torch.nn as neural_network

class LegalBiLSTM(neural_network.Module):
    def __init__(self, input_dimension = 16, hidden_dimension = 32, output_dimension = 3):
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