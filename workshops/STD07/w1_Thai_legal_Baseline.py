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
       placeholder_token = ""
    #    placeholder_token = f"__KW_{idx}__"
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

# การวัดค่า คาวมกำกวม (ambiguity) ของคำในบริบทกฎหมาย เทียบระหว่าง Dict และ Base+Regex
def calculate_baseline_ambiguity(text):
    matches = []
    for key_word in LEGAL_KEYWORD:
       for match in re.finditer(re.escape(key_word), text):
           if match:
               matches.append((match.start(), match.end(), key_word))
           # ตรวจสอบการทับซ้อนของคำที่เป็น Overlap
    overlaps = 0
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            if matches[i][0] < matches[j][1] and matches[i][1] > matches[j][0]: # เนื่องจากเป็น array 2มิติ ต้องใช้ index 0 และ 1 เพื่อเข้าถึงตำแหน่งเริ่มต้นและสิ้นสุดของคำที่จับได้
                overlaps += 1
                # break  # นับแค่ครั้งเดียวต่อคำที่ทับซ้อน
    return overlaps/len(matches) if matches else 0.0

# ทดสอบรัน
print(""*50)
sample_text = "คดีการละเมิดสิทธิบัตรและเครื่องหมายการค้าในประเทศไทยมีความซับซ้อนมากขึ้น"
tokens = legal_tokenizer(sample_text)
print(f"Original Text: {sample_text}")
print(f"Tokens: {tokens}")
print(f"Baseline Ambiguity Score: {calculate_baseline_ambiguity(sample_text):.2f}")
print(""*50)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
# Load WangchanBERTa model สำหรับการทำ Token Classification
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)

def berta_tokenizer(text):
    # ใช้ WangchanBERTa tokenizer
    tokenizer.tokenize(text)
    return [t.replace("▁", "") for t in tokenizer.tokenize(text) if t.replace("▁", "")]  # ลบสัญลักษณ์พิเศษที่ WangchanBERTa ใช้

# ทดสอบการวัด คำกำกวมด้วย WangchanBERTa tokenizer
def analyze_refined_ambiguity(text, legaal_keywords):
    text_tokens = berta_tokenizer(text)
    frag_scores = []
    for key_word in legaal_keywords:
        if key_word not in text:
            continue
        # เปรียบเทียบ keyword กับการ tokenize ใน context ของ text จริง
        # keyword ที่ "ไม่กำกวม" = ปรากฏเป็น token เดียวใน text
        # keyword ที่ "กำกวม" = ถูกกลืนเข้าไปใน token อื่น หรือถูกแตกเป็นหลายชิ้น
        if key_word in text_tokens:
            fragment_ratio = 1.0  # ตรงกับ token เดียว → ไม่กำกวม
        else:
            # ไม่เจอ keyword ตรงๆ ใน text tokens → ถูกรวม/แตก → กำกวม
            keyword_tokens = berta_tokenizer(key_word)
            fragment_ratio = max(2.0, len(keyword_tokens) + 1.0)
        frag_scores.append(fragment_ratio)
    if not frag_scores:
        return 0.0
    avg_frag_score = sum(frag_scores) / len(frag_scores) - 1
    return min(avg_frag_score, 1.0)

# ทดสอบรัน
print(""*50)
sample_text = "คดีการละเมิดสิทธิบัตรและเครื่องหมายการค้าในประเทศไทยมีความซับซ้อนมากขึ้น"
ambiguity_score = analyze_refined_ambiguity(sample_text, LEGAL_KEYWORD)
print(f"Original Text: {sample_text}")
print(f"Tokens: {berta_tokenizer(sample_text)}")
print(f"Refined Ambiguity Score: {ambiguity_score:.2f}")
print(""*50)




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