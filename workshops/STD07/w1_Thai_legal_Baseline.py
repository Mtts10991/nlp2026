import re
from pythainlp.tokenize import word_tokenize
from pythainlp.tokenize.attacut import AttacutTokenizer

LEGAL_KEYWORD = ["ละเมิด","ปลอมแปลง","เลียนแบบ","ทำซ้ำ","ดัดแปลง","สิทธิบัตร","การประดิษฐ์","ผังภูมิวงจร","ลิขสิทธิ์","วรรณกรรม","ศิลปกรรม","ดนตรีกรรม"]

def legal_tokenizer(text):
    # ใช้ AttacutTokenizer
   sorted_keywords = sorted(LEGAL_KEYWORD, key=len, reverse=True)
   placeholder = {}
   protected_text = text
   for idx, kw in enumerate(sorted_keywords):
       placeholder_token = f"__KW_{idx}__"
       placeholder[placeholder_token] = kw
       protected_text = protected_text.replace(kw, placeholder_token)

   # ใช้ AttacutTokenizer ในการ tokenize ข้อความที่ได้
   tokenizer = AttacutTokenizer()
   tokens = word_tokenize(protected_text)

   # แทนที่ token ที่เป็น keyword ด้วย keyword จริง
   legal_tokens = [placeholder.get(t, t) for t in tokens]

   return legal_tokens

# ทดสอบรัน
    # sample_text = "พบการทำซ้ำวรรณกรรมโดยไม่ได้รับอนุญาต"
    # tokens = legal_tokenizer(sample_text)
    # print(f"Original Text: {sample_text}")
    # print(f"Tokens: {tokens}")

#2 Context-aware Entity Extraction
def extract_legal_entities(text):
    # ใช้ AttacutTokenizer
   entities = []
   # จำลองหาความผิด และประเภทความผิด และ หาการกระทำผิดที่เกี่ยวข้อง
   if "สิทธิบัตร" in text:
       entities.append({"entity": "สิทธิบัตร", "ประเภทความผิด": "IP_TYPE", "value" : "PATENT", "confidence": 0.95})
   if "ละเมิด" in text:
       entities.append({"entity": "ละเมิด", "ประเภทความผิด": "ACTION", "value" : "INFRINGEMENT", "confidence": 0.95})
   return entities

# ทดสอบรัน
sample_text = "มีการละเมิดสิทธิบัตรในกรณีนี้เกิดขึ้นในพื้นที่ประเทศไทย"
found = extract_legal_entities(sample_text)

print(""*50)
print(f"Original Text: {sample_text}")
print(f"Found Entities: {found}")
print(""*50)

for entity in found:
    print(f"Entity: {entity['entity']}, Type: {entity['ประเภทความผิด']}, Value: {entity['value']}, Confidence: {entity['confidence']}")

#3 Feature Engineering (TF-IDF  Vectorization BASE)
from sklearn.feature_extraction.text import TfidfVectorizer
CORPUS = [
    "ละเมิดสิทธิบัตร เครื่องหมายการค้า",
    "การกระทำความผิด ลิขสิทธิ์ วรรณกรรม",
    "จำเลย ละเมิด ลิขสิทธิ์ ดนตรีกรรม",
    ]

# สร้าง Vactorizer และแปลงข้อความเป็น TF-IDF vectors
vectorizer = TfidfVectorizer(tokenizer=legal_tokenizer, token_pattern=None)  # ใช้ legal_tokenizer เป็น tokenizer และปิด token_pattern
tfidf_matrix = vectorizer.fit_transform(CORPUS)
print(""*50)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Feature Names:", vectorizer.get_feature_names_out())
print(f"Vactorized Matrix Simple (Doc 1):\n{tfidf_matrix[1].toarray()}")
print(""*50)

#4 Physical Gate Weight (จำลองการให้ความสำคัญกับคำที่เกี่ยวข้องกับกฎหมายมากขึ้น)
def compute_physical_gate_weight(entities):
    WEIGHT_BASE = 5.0 # น้ำหนักพื้นฐาน
    for entity in entities:
        if entity['value'] == "PATENT":
            WEIGHT_BASE += 2.0  # เพิ่มน้ำหนักสำหรับคำที่เป็น keyword
        if entity['value'] == "INFRINGEMENT":
            WEIGHT_BASE += 1.5  # เพิ่มน้ำหนักสำหรับประเภทความผิดที่เกี่ยวข้องกับ การละเมิด
    return min(WEIGHT_BASE, 10.0)  # จำกัดน้ำหนักสูงสุดที่ 10.0

# ทดสอบรัน
print(""*50)
weight = compute_physical_gate_weight(found)
print(f"Computed Physical Gate Weight: {weight:.2f} / 10")
print(f"Status : {'High' if weight > 7.0 else 'Moderate' if weight > 5.0 else 'Low'}")
print(""*50)