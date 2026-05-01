import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from collections import Counter
from statistics import mean

from w1_thai_legal_nlp import (
    ThaiIPEntityExtractor,
    ThaiLegalHierarchy,
    DATA_PATH,
)

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

LABEL_NAMES = data["metadata"]["label_names"]
samples = data["samples"]

extractor = ThaiIPEntityExtractor(use_context_aware=True)
hierarchy = ThaiLegalHierarchy()

# ================================================================
# Run extractor on first 10 docs (ตามโจทย์)
# ================================================================
print("=" * 70)
print("  ThaiIPEntityExtractor + Physics Gate Weight (first 10 docs)")
print("=" * 70)
for d in samples[:10]:
    entities = extractor.extract(d["text"])
    weight = hierarchy.compute_physics_gate_weight(entities)
    print(f"  Label = {d['label']}  weight = {weight:.2f}  "
          f"entities = {[e.entity_type for e in entities]}")

# ================================================================
# Run extractor on FULL dataset (สำหรับตอบคำถาม h, i)
# ================================================================
results = []
for d in samples:
    entities = extractor.extract(d["text"])
    weight = hierarchy.compute_physics_gate_weight(entities)
    results.append({
        "label": d["label"],
        "text": d["text"],
        "entities": entities,
        "weight": weight,
    })

# ================================================================
# (h) Entity types ที่พบบ่อยใน class 0 vs class 1
# W 1.3 H : entity ปะเภทใดพบบ่อยที่สุดใน class 0 แล้ว class 1 — เปรียบเทียบแล้วอธิบาย
# ================================================================
print("\n" + "=" * 70)
print("  (h) Entity types — Class 0 (ละเมิดสิทธิบัตร) vs Class 1 (ละเมิดลิขสิทธิ์)")
print("=" * 70)
for classId in [0, 1]:
    # 1) กรอง doc เฉพาะ class นี้
    docs_in_class = [r for r in results if r["label"] == classId]
    n_docs = len(docs_in_class)

    # 2) นับ entity types ในทุก doc ของ class
    counter = Counter()
    for r in docs_in_class:
        for e in r["entities"]:
            counter[e.entity_type] += 1

    # 3) print ผลลัพธ์
    print(f"\n  Class {classId} ({LABEL_NAMES[classId]}) — {n_docs} docs")
    for ent_type, cnt in counter.most_common():
        print(f"    {ent_type:<10} {cnt:>3}  (avg {cnt/n_docs:.2f}/doc)")


"""
======================================================================
  (h) Entity types — Class 0 (ละเมิดสิทธิบัตร) vs Class 1 (ละเมิดลิขสิทธิ์)
======================================================================

  Class 0 (ละเมิดสิทธิบัตร) — 40 docs
    IP_TYPE     40  (avg 1.00/doc)
    ACTION      16  (avg 0.40/doc)

  Class 1 (ละเมิดลิขสิทธิ์) — 20 docs
    ACTION      14  (avg 0.70/doc)
    IP_TYPE     10  (avg 0.50/doc)
"""

# ================================================================
# (i) Physics Gate Weight average — class 0 vs class 2
# W 1.3 I : Physics Gate Weight เฉลี่ยของ class 0 สูงกว่า class 2 หรีอไม่? — อธิบายว่าค่า weight นี้จะนำไปัใช้สั่ง IoT Sensor ได้อย่างไร
# ================================================================
print("\n" + "=" * 70)
print("  (i) Physics Gate Weight average per class")
print("=" * 70)
for cls in [0, 1, 2]:
    weights = [r["weight"] for r in results if r["label"] == cls]
    avg = mean(weights) if weights else 0.0
    nonzero = sum(1 for w in weights if w > 0)
    print(f"  Class {cls} ({LABEL_NAMES[cls]:<15}) "
          f"weight avg = {avg:.3f}  weight max = {max(weights):.2f}  "
          f"docs with weight > 0 = {nonzero} / {len(weights)}")

print(f"\n   ทุก class ได้ weight=0 ด้วย 2 เหตุผล:")
print(f"")
print(f"   ─── เหตุผล 1: Dataset ไม่มี doc ไหนอ้าง 'มาตรา X' เลย ───")
print(f"     • text ใน data_set.py ทั้ง 66 ตัวอย่างไม่มีคำว่า 'มาตรา' สักตัว")
print(f"     • เช่น 'จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345'")
print(f"     • Extractor มี pattern 'มาตรา\\s*\\d+' ที่ต้อง match ถึงจะสร้าง")
print(f"       STATUTE entity ได้ → เมื่อไม่มีในข้อความ ก็ไม่มี STATUTE entity")
print(f"     • compute_physics_gate_weight() loop หา STATUTE ไม่เจอ → return 0")
print(f"")
print(f"   ─── เหตุผล 2: มี bug ใน w1_thai_legal_nlp.py:480 ───")
print(f"     โค้ดที่ผิด:")
print(f"       if ent.entity_type == 'STATUTE' and ent.law_reference:")
print(f"                                           ↑ field นี้เป็น None เสมอ")
print(f"")
print(f"     trace ทีละ step (text='ละเมิดสิทธิบัตรตามมาตรา 77'):")
print(f"")
print(f"     Step 1: extractor เจอ 'มาตรา 77' ที่ position 16")
print(f"             → สร้าง STATUTE entity, value='มาตรา 77'")
print(f"             → ตอนสร้างเรียก _find_statute(text, 16) เพื่อ set law_reference")
print(f"")
print(f"     Step 2: _find_statute(text, 16) ค้น regex ใน text[:16]")
print(f"             text[:16] = 'ละเมิดสิทธิบัตรตาม'  ← ไม่มี 'มาตรา' อยู่ก่อน pos 16")
print(f"             → return None")
print(f"             → entity.law_reference = None")
print(f"")
print(f"     Step 3: compute_physics_gate_weight() เช็คเงื่อนไข")
print(f"             ent.entity_type == 'STATUTE'  → True")
print(f"             ent.law_reference             → None (= falsy)")
print(f"             → เงื่อนไข and เป็น False → ข้าม entity นี้")
print(f"             → total weight = 0")
print(f"")
print(f"     ทำไมถึงเป็น bug:")
print(f"       law_reference ออกแบบมาตอบ 'entity นี้อ้างอิงมาตราไหน'")
print(f"       เหมาะกับ IP_TYPE/PENALTY ที่ต้องชี้ไปยังมาตราอื่น")
print(f"       แต่ STATUTE entity เอง — value='มาตรา 77' ก็คือ reference อยู่แล้ว")
print(f"       ไม่ต้องไปค้นหา 'มาตรา' อื่นใน text อีก")
print(f"")
print(f"     วิธีแก้: เปลี่ยนเงื่อนไขเป็น ent.value (ดู fixed_weight() ด้านล่าง)")

print(f"\n  ▶ Sanity check (สภาพปัจจุบัน — bug ยังอยู่):")
synthetic = [
    "ละเมิดสิทธิบัตรตามมาตรา 77",
    "ทำซ้ำงานละเมิดลิขสิทธิ์ตามมาตรา 70",
    "บริษัทได้รับอนุญาตอย่างถูกต้องตามสัญญา",
]
for text in synthetic:
    ents = extractor.extract(text)
    w = hierarchy.compute_physics_gate_weight(ents)
    print(f"     weight={w:5.2f}  text='{text}'")

print(f"\n  ▶ ถ้าแก้ bug (ใช้ ent.value แทน ent.law_reference):")
def fixed_weight(entities):
    total = 0.0
    for ent in entities:
        if ent.entity_type == "STATUTE":
            ip_types = [e.value for e in entities if e.entity_type == "IP_TYPE"]
            if ip_types:
                cls = hierarchy.classify_offense(ent.value, ip_types[0])
                w = cls.get("severity", 0) * ent.confidence
                w *= 0.8 if cls.get("can_settle") else 1.2
                total += w
    return min(10.0, total)

for text in synthetic:
    ents = extractor.extract(text)
    w = fixed_weight(ents)
    print(f"     weight={w:5.2f}  text='{text}'")

"""
======================================================================
  (i) Physics Gate Weight average per class
======================================================================
  Class 0 (ละเมิดสิทธิบัตร) weight avg = 0.000  weight max = 0.00  docs with weight > 0 = 0 / 40
  Class 1 (ละเมิดลิขสิทธิ์) weight avg = 0.000  weight max = 0.00  docs with weight > 0 = 0 / 20
  Class 2 (ไม่ละเมิด      ) weight avg = 0.000  weight max = 0.00  docs with weight > 0 = 0 / 6

   ทุก class ได้ weight=0 ด้วย 2 เหตุผล:

   ─── เหตุผล 1: Dataset ไม่มี doc ไหนอ้าง 'มาตรา X' เลย ───
     • text ใน data_set.py ทั้ง 66 ตัวอย่างไม่มีคำว่า 'มาตรา' สักตัว
     • เช่น 'จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345'
     • Extractor มี pattern 'มาตรา\s*\d+' ที่ต้อง match ถึงจะสร้าง
       STATUTE entity ได้ → เมื่อไม่มีในข้อความ ก็ไม่มี STATUTE entity
     • compute_physics_gate_weight() loop หา STATUTE ไม่เจอ → return 0

   ─── เหตุผล 2: มี bug ใน w1_thai_legal_nlp.py:480 ───
     โค้ดที่ผิด:
       if ent.entity_type == 'STATUTE' and ent.law_reference:
                                           ↑ field นี้เป็น None เสมอ

     trace ทีละ step (text='ละเมิดสิทธิบัตรตามมาตรา 77'):

     Step 1: extractor เจอ 'มาตรา 77' ที่ position 16
             → สร้าง STATUTE entity, value='มาตรา 77'
             → ตอนสร้างเรียก _find_statute(text, 16) เพื่อ set law_reference

     Step 2: _find_statute(text, 16) ค้น regex ใน text[:16]
             text[:16] = 'ละเมิดสิทธิบัตรตาม'  ← ไม่มี 'มาตรา' อยู่ก่อน pos 16
             → return None
             → entity.law_reference = None

     Step 3: compute_physics_gate_weight() เช็คเงื่อนไข
             ent.entity_type == 'STATUTE'  → True
             ent.law_reference             → None (= falsy)
             → เงื่อนไข and เป็น False → ข้าม entity นี้
             → total weight = 0

     ทำไมถึงเป็น bug:
       law_reference ออกแบบมาตอบ 'entity นี้อ้างอิงมาตราไหน'
       เหมาะกับ IP_TYPE/PENALTY ที่ต้องชี้ไปยังมาตราอื่น
       แต่ STATUTE entity เอง — value='มาตรา 77' ก็คือ reference อยู่แล้ว
       ไม่ต้องไปค้นหา 'มาตรา' อื่นใน text อีก

     วิธีแก้: เปลี่ยนเงื่อนไขเป็น ent.value (ดู fixed_weight() ด้านล่าง)

  ▶ Sanity check (สภาพปัจจุบัน — bug ยังอยู่):
     weight= 0.00  text='ละเมิดสิทธิบัตรตามมาตรา 77'
     weight= 0.00  text='ทำซ้ำงานละเมิดลิขสิทธิ์ตามมาตรา 70'
     weight= 0.00  text='บริษัทได้รับอนุญาตอย่างถูกต้องตามสัญญา'

  ▶ ถ้าแก้ bug (ใช้ ent.value แทน ent.law_reference):
     weight= 5.60  text='ละเมิดสิทธิบัตรตามมาตรา 77'
     weight= 9.60  text='ทำซ้ำงานละเมิดลิขสิทธิ์ตามมาตรา 70'
     weight= 0.00  text='บริษัทได้รับอนุญาตอย่างถูกต้องตามสัญญา'
"""

# ================================================================
# (j) Context-aware Confidence — ตัวอย่างที่ confidence เพิ่มขึ้นจาก context signal
# W 1.3 J : Context-aware Confidence Scoring ทำงานอย่างไร — ยกตัวอย่าง 1 ประโยคที่ confidence เพิ่มขึ้นจาก context_signals
# ================================================================
print("\n" + "=" * 70)
print("  (j) Context-aware Confidence Scoring — ตัวอย่างที่ context boost confidence")
print("=" * 70)

# หา doc ที่มี entity ซึ่ง context_signals มี "_present" (= context boost ทำงาน)
example = None
for r in results:
    for e in r["entities"]:
        if any(sig.endswith("_present") for sig in e.context_signals):
            example = (r, e)
            break
    if example:
        break

if example:
    r, e = example
    print(f"\n  Doc text  : {r['text']}")
    print(f"  Label     : {LABEL_NAMES[r['label']]}")
    print(f"  Entity    : [{e.entity_type}] '{e.value}'")
    print(f"  Confidence: {e.confidence:.3f}")
    print(f"  Signals   : {e.context_signals}")
    print(f"\n  อธิบาย: entity '{e.value}' (type={e.entity_type}) ได้ base confidence")
    print(f"  จากการ match pattern แล้วถูกปรับด้วย context signals ที่เจอใน window")
    print(f"  รอบ ๆ entity (ดู ContextAwareConfidenceScorer.compute_confidence)")
    print(f"  signal ที่ลงท้าย '_present' = boost (×1.3 หรือ ×1.2)")
    print(f"  signal ที่ลงท้าย '_absent'  = penalty (×0.7 หรือ ×0.85)")

"""
======================================================================
  (j) Context-aware Confidence Scoring — ตัวอย่างที่ context boost confidence
======================================================================

  Doc text  : จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345
  Label     : ละเมิดสิทธิบัตร
  Entity    : [IP_TYPE] 'สิทธิบัตร'
  Confidence: 0.541
  Signals   : ['PATENT_CONTEXT_present', 'COPYRIGHT_CONTEXT_absent', 'TRADEMARK_CONTEXT_absent']

  อธิบาย: entity 'สิทธิบัตร' (type=IP_TYPE) ได้ base confidence
  จากการ match pattern แล้วถูกปรับด้วย context signals ที่เจอใน window
  รอบ ๆ entity (ดู ContextAwareConfidenceScorer.compute_confidence)
  signal ที่ลงท้าย '_present' = boost (×1.3 หรือ ×1.2)
  signal ที่ลงท้าย '_absent'  = penalty (×0.7 หรือ ×0.85)
"""
