import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np

from w1_thai_legal_nlp import W1Pipeline, DATA_PATH

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

LABEL_NAMES = data["metadata"]["label_names"]
texts = [s["text"] for s in data["samples"]]
labels = [s["label"] for s in data["samples"]]

w1 = W1Pipeline(max_features=50, oov_strategy="unk_token", evaluate_coverage=True)
X = w1.fit_transform(texts) # X = INPUT
#===============================================================================
# คำตอบ 1.2: D แสดง shape ของ TF-IDF matrix X — อธิบายว่าแต่ละ dimension หมายถึงอะไร
#===============================================================================

print("\n" + "=" * 70)
print("  (d) Shape of TF-IDF matrix X")
print("=" * 70)
print(f"  X.shape = {X.shape}")
print(f"    - rows ({X.shape[0]}) = number of documents in corpus")
print(f"    - cols ({X.shape[1]}) = vocabulary size (max_features = 50)")
print(f"  Each row X[i] is a vector of doc i in 50-D vocabulary space")
print(f"  Each X[i,j] = normalized TF-IDF weight of term j in doc i")

"""
======================================================================
  (d) Shape of TF-IDF matrix X
======================================================================
  X.shape = (66, 50)
    - rows (66) = number of documents in corpus
    - cols (50) = vocabulary size (max_features = 50)
  Each row X[i] is a vector of doc i in 50-D vocabulary space
  Each X[i,j] = normalized TF-IDF weight of term j in doc i
"""
#===============================================================================
# คำตอบ 1.2: E หา top-5 term ที่มีน้ำหนัก TF-IDF สูงสุดใน document ข้อง class 0, 1, 2 (อย่างละ 1 ตัวอย่าง)
#===============================================================================

print("\n" + "=" * 70)
print("  (e) Top-5 TF-IDF terms per class (1 example each)")
print("=" * 70)
for cls in [0, 1, 2]:
    idx = next(i for i, lb in enumerate(labels) if lb == cls)
    top = w1.tfidf.get_top_terms(X[idx], n=5)
    print(f"\n  Class {cls} ({LABEL_NAMES[cls]}) — idx : {idx}")
    print(f"    Text: {texts[idx]}")
    print(f"    Top-5 terms:")
    for term, weight in top:
        print(f"      {weight:.4f}  {term}")

"""
======================================================================
  (e) Top-5 TF-IDF terms per class (1 example each)
======================================================================

  Class 0 (ละเมิดสิทธิบัตร) — idx : 0
    Text: จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345
    Top-5 terms:
      0.5187  12345
      0.5187  จำเลยผลิตสินค้าที่เลียนแบบ
      0.5187  เลขที่
      0.4390  สิทธิบัตรการประดิษฐ์

  Class 1 (ละเมิดลิขสิทธิ์) — idx : 40
    Text: จำเลยทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไม่ได้รับอนุญาต
    Top-5 terms:
      0.7845  โดยไม่ได้รับอนุญาต
      0.5197  ลิขสิทธิ์
      0.3383  [UNK]

  Class 2 (ไม่ละเมิด) — idx : 60
    Text: บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้องตามสัญญา
    Top-5 terms:
      1.0000  [UNK]
"""
#===============================================================================
# คำตอบ 1.2: F แสดง OOV Report — ระบุจำนวน OOV tokens และ sample 5 คำ พร้อมเสนอวิธิแก้ไข
#===============================================================================

print("\n" + "=" * 70)
print("  (f) OOV Report")
print("=" * 70)
oov = w1.get_oov_report()
print(f"  Total OOV tokens : {oov['total_oov']}")
print(f"  Strategy         : {oov['strategy']}")
print(f"  Sample (5 words) :")
for word in oov['sample_oov'][:5]:
    print(f"    - {word}")
print(f"\n  วิธีแก้:")
print(f"    1. เพิ่ม max_features (ตอนนี้ 50 → ลอง 100 หรือ 200) เพื่อเก็บคำหายากเข้า vocab")
print(f"    2. เพิ่ม LEGAL_COMPOUNDS ให้ครอบคลุม domain term ที่ตกหล่น")
print(f"    3. ใช้ Thai word segmenter (PyThaiNLP) แทน naive tokenizer เพราะ OOV ส่วนใหญ่เป็นก้อน Thai ยาว ๆ ที่ไม่ถูกตัดคำ")

"""
======================================================================
  (f) OOV Report
======================================================================
  Total OOV tokens : 42
  Strategy         : unk_token
  Sample (5 words) :
    - ที่จดสิทธิบัตรไว้
    - จำเลยผลิตวัสดุก่อสร้างโดยใช้สูตรซีเมนต์ที่ได้รับสิทธิบัตร
    - ผู้ต้องหานำเข้ายาชีววัตถุที่ละเมิดสิทธิบัตรของ
    - บริษัทจำเลยผลิตตัวเก็บประจุโดยใช้วัสดุไดอิเล็กตริกตามสิทธิบัตร
    - จำเลยทำซ้ำโปรแกรมคอมพิวเตอร์มี

  วิธีแก้:
    1. เพิ่ม max_features (ตอนนี้ 50 → ลอง 100 หรือ 200) เพื่อเก็บคำหายากเข้า vocab
    2. เพิ่ม LEGAL_COMPOUNDS ให้ครอบคลุม domain term ที่ตกหล่น
    3. ใช้ Thai word segmenter (PyThaiNLP) แทน naive tokenizer เพราะ OOV ส่วนใหญ่เป็นก้อน Thai ยาว ๆ ที่ไม่ถูกตัดคำ
"""
#===============================================================================
# คำตอบ 1.2: G เรียก w1.print_professional_report(texts) แล้วอธิบายผล Coverage ที่ได้
#===============================================================================

print("\n" + "=" * 70)
print("  (g) Professional Report")
print("=" * 70)
w1.print_professional_report(texts)

print("\n" + "─" * 70)
print("  อธิบาย Coverage:")
print("─" * 70)
cov = w1.get_coverage_report()
if cov:
    pct = cov['overall']['percentage']
    covered = cov['overall']['covered']
    total = cov['overall']['total']
    print(f"  Overall = {covered}/{total} = {pct:.1f}%")
    print(f"  → vocab ของ TF-IDF ({w1.n_features} terms) ครอบคลุมศัพท์กฎหมาย")
    print(f"     ใน LEGAL_GLOSSARY ได้กี่ % โดยแบ่ง 5 หมวด: PATENT, COPYRIGHT,")
    print(f"     TRADEMARK, ENFORCEMENT, PROCEDURE")
    print(f"  → ถ้า % ต่ำ แปลว่า model จะอ่านศัพท์กฎหมายไม่ครบ")
    print(f"     ทำนายผิดง่าย — แก้ด้วยการเพิ่ม max_features หรือใช้ tokenizer ที่ดีกว่า")


"""
======================================================================
  (g) Professional Report
======================================================================

██████████████████████████████████████████████████████████████████████
  PROFESSIONAL QUALITY REPORT — W1 Pipeline v3
  For PhD Thesis: Physics-Governed IoT Framework for IP Law Enforcement
██████████████████████████████████████████████████████████████████████

  📚 LEGAL VOCABULARY COVERAGE:
     Overall: 3.9%

  🔤 OUT-OF-VOCABULARY (OOV) STATISTICS:
     Total OOV tokens: 42
     Sample OOV: ที่จดสิทธิบัตรไว้, จำเลยผลิตวัสดุก่อสร้างโดยใช้สูตรซีเมนต์ที่ได้รับสิทธิบัตร, ผู้ต้องหานำเข้ายาชีววัตถุที่ละเมิดสิทธิบัตรของ, บริษัทจำเลยผลิตตัวเก็บประจุโดยใช้วัสดุไดอิเล็กตริกตามสิทธิบัตร, จำเลยทำซ้ำโปรแกรมคอมพิวเตอร์มี

  ⚠️  AMBIGUITY ANALYSIS:
     Ambiguity rate: 72.64%
     ❌ อัตราความกำกวมสูง ต้องปรับปรุง tokenizer หรือใช้ W4 (WangchanBERTa)

  💡 RECOMMENDATIONS FOR IMPROVEMENT:
     1. High ambiguity rate detected — consider using W4 (WangchanBERTa)
     2. Coverage 3.9% < 80% — increase max_features
     3. OOV tokens detected (42) — expand LEGAL_COMPOUNDS

██████████████████████████████████████████████████████████████████████

──────────────────────────────────────────────────────────────────────
  อธิบาย Coverage:
──────────────────────────────────────────────────────────────────────
  Overall = 2/51 = 3.9%
  → vocab ของ TF-IDF (50 terms) ครอบคลุมศัพท์กฎหมาย
     ใน LEGAL_GLOSSARY ได้กี่ % โดยแบ่ง 5 หมวด: PATENT, COPYRIGHT,
     TRADEMARK, ENFORCEMENT, PROCEDURE
  → ถ้า % ต่ำ แปลว่า model จะอ่านศัพท์กฎหมายไม่ครบ
     ทำนายผิดง่าย — แก้ด้วยการเพิ่ม max_features หรือใช้ tokenizer ที่ดีกว่า
"""