import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import json
import random
import importlib.util
from contextlib import redirect_stdout
from pathlib import Path

# ===== Patch matplotlib ก่อน import w2 source (ที่จะ plt.show ตอน load) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *_args, **_kwargs: None

# ===== Import จากไฟล์ต้นฉบับที่มี space ในชื่อ — ใช้ importlib =====
SRC_PATH = Path(__file__).parent / "w2 Thai IP Legal NLP.py"
spec = importlib.util.spec_from_file_location("w2_src", SRC_PATH)
w2_src = importlib.util.module_from_spec(spec)
# source file รัน training/plot ตอนโหลด และ class mismatch จะ raise ValueError
# แต่ LEGAL_SYNONYMS + augment_legal_text ถูกประกาศก่อน error → ใช้ได้
with redirect_stdout(io.StringIO()):
    try:
        spec.loader.exec_module(w2_src)
    except Exception:
        pass

LEGAL_SYNONYMS = w2_src.LEGAL_SYNONYMS
augment_legal_text = w2_src.augment_legal_text

# ===== Load dataset =====
DATA_PATH = Path(__file__).parent / "data" / "processed" / "thai_ip_dataset.json"
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
LABEL_NAMES = data["metadata"]["label_names"]
samples = data["samples"]

class0_texts = [s["text"] for s in samples if s["label"] == 0]
random.seed(42)
augmented = [augment_legal_text(t) for t in class0_texts]

# ================================================================
# (k) แสดง original → augmented อย่างน้อย 5 คู่
# W 2.1 : K แสดงคู่originl  augmented → อย่างน้อย 5 คู่ —  ระบุว่าคำใดถูกต้อง
# ================================================================
print("=" * 70)
print("  (k) Original → Augmented (5 pairs จาก class 0)")
print("=" * 70)

shown = 0
for orig, aug in zip(class0_texts, augmented):
    if orig != aug:
        print(f"\n  Original  : {orig}")
        print(f"  Augmented : {aug}")
        shown += 1
        if shown >= 5:
            break

if shown == 0:
    print("\n  ⚠️  ไม่มีคู่ไหนเปลี่ยนเลย — augment_legal_text() ไม่ทำงานกับ dataset นี้")

"""
======================================================================
  (k) Original → Augmented (5 pairs จาก class 0)
======================================================================

  ⚠️  ไม่มีคู่ไหนเปลี่ยนเลย — augment_legal_text() ไม่ทำงานกับ dataset นี้
"""
# ================================================================
# (l) ประโยคที่ไม่มีคีย์ใน LEGAL_SYNONYMS — augmented = original ?
# W 2.1 : L ประโยคที่ไม่มีคำใน LEGAL_SYNONYMS จะเกิดอะไรขึ้น — augmented text เหมือน original หรือไม่
# ================================================================
print("\n" + "=" * 70)
print("  (l) ประโยคที่ไม่มี keyword ใน LEGAL_SYNONYMS")
print("=" * 70)

keywords = list(LEGAL_SYNONYMS.keys())
print(f"  Keywords: {keywords}")

# ตรวจสอบทั้ง class 0
unchanged_count = sum(1 for o, a in zip(class0_texts, augmented) if o == a)
no_keyword_count = sum(1 for t in class0_texts if not any(kw in t for kw in keywords))

print(f"\n  จาก {len(class0_texts)} ประโยคใน class 0:")
print(f"    augmented = original   : {unchanged_count} ประโยค")
print(f"    ไม่มี keyword ในประโยค : {no_keyword_count} ประโยค")

print(f"\n  ⚠️  ปัญหาที่พบ: ทุกประโยคไม่ถูกแทนเลย ทั้ง ๆ ที่หลายประโยคมี 'ละเมิด'")
print(f"")
print(f"  ▶ Trace ทำไม augment_legal_text() ไม่ทำงานกับ dataset:")
print(f"")
print(f"    โค้ดต้นฉบับ:")
print(f"      def augment_legal_text(text):")
print(f"          words = text.split()         ← split ตาม whitespace")
print(f"          for i, word in enumerate(words):")
print(f"              if word in LEGAL_SYNONYMS:   ← เทียบ word == keyword (exact)")
print(f"                  new_words[i] = ...")
print(f"")
print(f"    ทดสอบกับ doc แรก:")
sample_text = class0_texts[0]
print(f"      text  = '{sample_text}'")
print(f"      split = {sample_text.split()}")
print(f"      → split() ได้ {len(sample_text.split())} token (เพราะภาษาไทยไม่มี space)")
print(f"      → ทุก token ไม่ตรงกับ keyword 'ละเมิด' / 'จำหน่าย' / 'ปลอมแปลง'")
print(f"      → ไม่มีอะไรถูกแทน → return text เดิม")
print(f"")
print(f"    เปรียบเทียบกับ demo ใน source file (ที่ทำงานได้):")
print(f"      original = 'จำเลย ละเมิด และ จำหน่าย สินค้า'  ← มี space")
print(f"      split    = ['จำเลย', 'ละเมิด', 'และ', 'จำหน่าย', 'สินค้า']")
print(f"      → 'ละเมิด' ตรง keyword → ถูกแทน ✓")
print(f"")
print(f"  สรุป: augment_legal_text() ออกแบบมาสำหรับ text ที่ pre-tokenized แล้ว")
print(f"        ใช้กับ dataset ของเราตรง ๆ ไม่ได้ ต้องตัดคำก่อน")

"""
======================================================================
  (l) ประโยคที่ไม่มี keyword ใน LEGAL_SYNONYMS
======================================================================
  Keywords: ['ละเมิด', 'จำหน่าย', 'ปลอมแปลง']

  จาก 40 ประโยคใน class 0:
    augmented = original   : 40 ประโยค
    ไม่มี keyword ในประโยค : 23 ประโยค

  ⚠️  ปัญหาที่พบ: ทุกประโยคไม่ถูกแทนเลย ทั้ง ๆ ที่หลายประโยคมี 'ละเมิด'

  ▶ Trace ทำไม augment_legal_text() ไม่ทำงานกับ dataset:

    โค้ดต้นฉบับ:
      def augment_legal_text(text):
          words = text.split()         ← split ตาม whitespace
          for i, word in enumerate(words):
              if word in LEGAL_SYNONYMS:   ← เทียบ word == keyword (exact)
                  new_words[i] = ...

    ทดสอบกับ doc แรก:
      text  = 'จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345'
      split = ['จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่', '12345']
      → split() ได้ 2 token (เพราะภาษาไทยไม่มี space)
      → ทุก token ไม่ตรงกับ keyword 'ละเมิด' / 'จำหน่าย' / 'ปลอมแปลง'
      → ไม่มีอะไรถูกแทน → return text เดิม

    เปรียบเทียบกับ demo ใน source file (ที่ทำงานได้):
      original = 'จำเลย ละเมิด และ จำหน่าย สินค้า'  ← มี space
      split    = ['จำเลย', 'ละเมิด', 'และ', 'จำหน่าย', 'สินค้า']
      → 'ละเมิด' ตรง keyword → ถูกแทน ✓

  สรุป: augment_legal_text() ออกแบบมาสำหรับ text ที่ pre-tokenized แล้ว
        ใช้กับ dataset ของเราตรง ๆ ไม่ได้ ต้องตัดคำก่อน
"""

# ================================================================
# (m) เพิ่ม synonym ใหม่อย่างน้อย 2 คำ + แก้ปัญหาให้ใช้กับ dataset ได้
# W 2.1 : M เพิ่ม synonym ใหม่อย่างน้อย 2 คำที่เหมาะกับ dataset พร้อมให้เหตุผล
# ================================================================
print("\n" + "=" * 70)
print("  (m) เพิ่ม synonym ใหม่ + แก้ให้ใช้กับ Thai text ได้")
print("=" * 70)

# 2 keyword ใหม่ + เพิ่ม "ทำซ้ำ" สำหรับ class 1
EXTRA_SYNONYMS = {
    "ผลิต":   ["สร้าง", "ประกอบ", "ทำ"],         # ปรากฏใน 30+ doc ของ class 0
    "นำเข้า": ["สั่งเข้า", "นำมาจากต่างประเทศ"], # ปรากฏใน 10+ doc
    "ทำซ้ำ":  ["คัดลอก", "ผลิตซ้ำ"],             # ปรากฏใน class 1
}
EXTENDED = {**LEGAL_SYNONYMS, **EXTRA_SYNONYMS}
print(f"  เพิ่ม 3 keyword ใหม่ที่เหมาะกับ dataset:")
for kw, syns in EXTRA_SYNONYMS.items():
    count = sum(1 for t in class0_texts if kw in t)
    print(f"    '{kw}' → {syns}  (พบใน {count} ประโยค class 0)")

# แก้ปัญหา split() ด้วย substring replace
def augment_substring(text, synonyms=EXTENDED):
    """แก้จาก augment_legal_text ของ source — ใช้ str.replace แทน split()"""
    out = text
    for keyword, options in synonyms.items():
        if keyword in out:
            out = out.replace(keyword, random.choice(options), 1)
    return out

print(f"\n  ▶ ทดสอบ augment_substring() (ใช้ str.replace) กับ class 0:")
random.seed(42)
fixed_aug = [augment_substring(t) for t in class0_texts]
changed = sum(1 for o, a in zip(class0_texts, fixed_aug) if o != a)
print(f"    {changed} / {len(class0_texts)} ประโยคถูก augment สำเร็จ "
      f"(เดิม 0 ด้วย augment_legal_text)")

print(f"\n  ▶ ตัวอย่าง 5 คู่ที่ augment สำเร็จ:")
shown = 0
for orig, aug in zip(class0_texts, fixed_aug):
    if orig != aug:
        print(f"\n    Original  : {orig}")
        print(f"    Augmented : {aug}")
        shown += 1
        if shown >= 5:
            break

"""
======================================================================
  (m) เพิ่ม synonym ใหม่ + แก้ให้ใช้กับ Thai text ได้
======================================================================
  เพิ่ม 3 keyword ใหม่ที่เหมาะกับ dataset:
    'ผลิต' → ['สร้าง', 'ประกอบ', 'ทำ']  (พบใน 27 ประโยค class 0)
    'นำเข้า' → ['สั่งเข้า', 'นำมาจากต่างประเทศ']  (พบใน 7 ประโยค class 0)
    'ทำซ้ำ' → ['คัดลอก', 'ผลิตซ้ำ']  (พบใน 2 ประโยค class 0)

  ▶ ทดสอบ augment_substring() (ใช้ str.replace) กับ class 0:
    33 / 40 ประโยคถูก augment สำเร็จ (เดิม 0 ด้วย augment_legal_text)

  ▶ ตัวอย่าง 5 คู่ที่ augment สำเร็จ:

    Original  : จำเลยผลิตสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345
    Augmented : จำเลยทำสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์เลขที่ 12345

    Original  : ผู้ต้องหานำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ
    Augmented : ผู้ต้องหาสั่งเข้าชิ้นส่วนที่ฝ่าฝืนสิทธิบัตรจากต่างประเทศ

    Original  : บริษัทจำเลยผลิตยาสามัญโดยละเมิดสิทธิบัตรยาต้นแบบ
    Augmented : บริษัทจำเลยประกอบยาสามัญโดยล่วงสิทธิสิทธิบัตรยาต้นแบบ

    Original  : ผู้ต้องหาผลิตอุปกรณ์อิเล็กทรอนิกส์เลียนแบบสิทธิบัตรการประดิษฐ์
    Augmented : ผู้ต้องหาสร้างอุปกรณ์อิเล็กทรอนิกส์เลียนแบบสิทธิบัตรการประดิษฐ์

    Original  : จำเลยขายสินค้าปลอมแปลงที่ใช้กระบวนการผลิตตามสิทธิบัตร
    Augmented : จำเลยขายสินค้าทำเทียมที่ใช้กระบวนการสร้างตามสิทธิบัตร
"""
