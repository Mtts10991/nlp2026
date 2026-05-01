import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from w1_thai_legal_nlp import ThaiLegalTokenizer

tok = ThaiLegalTokenizer()

sentences = [
    "จำเลยผลิตและจำหน่ายสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์",
    "บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ",
    "ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไม่ได้รับอนุญาต",
]

#===============================================================================
# W1.1 : A แสดง tokens ที่ได้จากแต่ละประโยค — ระบุว่า LEGAL_COMPOUND ใดถูกรักษาเป็นคำ คำเดียวไปแล้ว
#===============================================================================

print("=" * 70)
print("  (a) Tokens from each sentence + LEGAL_COMPOUND that was protected")
print("=" * 70)
for i, s in enumerate(sentences, 1):
    tokens = tok.tokenize(s, remove_stopwords=False)
    matched = [c for c in tok.LEGAL_COMPOUNDS if c in s]
    print(f"\nSentence {i}: {s}")
    print(f"  Tokens          : {tokens}")
    print(f"  LEGAL_COMPOUNDS : {matched}")

"""
======================================================================
  (a) Tokens from each sentence + LEGAL_COMPOUND that was protected
======================================================================

Sentence 1: จำเลยผลิตและจำหน่ายสินค้าที่เลียนแบบสิทธิบัตรการประดิษฐ์
  Tokens          : ['จำเลยผลิตและจำหน่ายสินค้าที่เลียนแบบ', 'สิทธิบัตรการประดิษฐ์']
  LEGAL_COMPOUNDS : ['สิทธิบัตรการประดิษฐ์']

Sentence 2: บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ
  Tokens          : ['บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ']
  LEGAL_COMPOUNDS : []

Sentence 3: ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มีลิขสิทธิ์โดยไม่ได้รับอนุญาต
  Tokens          : ['ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มี', 'ลิขสิทธิ์', 'โดยไม่ได้รับอนุญาต']
  LEGAL_COMPOUNDS : ['ลิขสิทธิ์']
"""

#===============================================================================
# W1.1 : B ทดลองเพิ่ม remove_stopwords=True แล้วแสดงผลที่เปลียนไป — อธิบายว่า stopword ใดถูกตัดออก
#===============================================================================

print("\n" + "=" * 70)
print("  (b) With remove_stopwords=True")
print("=" * 70)
for i, s in enumerate(sentences, 1):
    before = tok.tokenize(s, remove_stopwords=False)
    after = tok.tokenize(s, remove_stopwords=True)
    removed = [t for t in before if t not in after]
    print(f"\nSentence {i}:")
    print(f"  Before (False) : {before}")
    print(f"  After (True)  : {after}")
    print(f"  Removed: {removed}")

"""
======================================================================
  (b) With remove_stopwords=True
======================================================================

Sentence 1:
  Before (False) : ['จำเลยผลิตและจำหน่ายสินค้าที่เลียนแบบ', 'สิทธิบัตรการประดิษฐ์']
  After (True)  : ['จำเลยผลิตและจำหน่ายสินค้าที่เลียนแบบ', 'สิทธิบัตรการประดิษฐ์']
  Removed: []

Sentence 2:
  Before (False) : ['บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ']
  After (True)  : ['บริษัทนำเข้าชิ้นส่วนที่ละเมิดสิทธิบัตรจากต่างประเทศ']
  Removed: []

Sentence 3:
  Before (False) : ['ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มี', 'ลิขสิทธิ์', 'โดยไม่ได้รับอนุญาต']
  After (True)  : ['ผู้ต้องหาทำซ้ำโปรแกรมคอมพิวเตอร์มี', 'ลิขสิทธิ์', 'โดยไม่ได้รับอนุญาต']
  Removed: []
"""

#===============================================================================
# W1.1 : C อธิบายว่าทำไม LEGAL_COMPOUND ถึงต้องเรียงจากยาวไปสั้น (sorted by len, reverse=True)
#===============================================================================

print("\n" + "=" * 70)
print("  (c) Why sort LEGAL_COMPOUNDS by length DESC?")
print("=" * 70)
print(f"\n  self.compounds (sorted longest first):")
for c in tok.compounds[:8]:
    print(f"    {len(c):2d} chars  {c}")

demo = "การละเมิดสิทธิบัตรการประดิษฐ์โดยจำเลย"
print(f"\n  Demo text: {demo}")
print(f"  Contains BOTH 'สิทธิบัตร' (9 chars) and 'สิทธิบัตรการประดิษฐ์' (20 chars)")

class WrongOrder(ThaiLegalTokenizer):
    def __init__(self):
        super().__init__()
        self.compounds = sorted(self.LEGAL_COMPOUNDS, key=len)

wrong = WrongOrder()
print(f"\n  Correct order (long→short): {tok.tokenize(demo)}")
print(f"  Wrong order   (short→long): {wrong.tokenize(demo)}")

"""
======================================================================
  (c) Why sort LEGAL_COMPOUNDS by length DESC?
======================================================================

  self.compounds (sorted longest first):
    20 chars  สิทธิบัตรการประดิษฐ์
    20 chars  ศาลทรัพย์สินทางปัญญา
    19 chars  การประดิษฐ์ขึ้นใหม่
    18 chars  พนักงานเจ้าหน้าที่
    17 chars  เครื่องหมายการค้า
    17 chars  ทรัพย์สินทางปัญญา
    17 chars  เครื่องหมายบริการ
    17 chars  เครื่องหมายรับรอง

  Demo text: การละเมิดสิทธิบัตรการประดิษฐ์โดยจำเลย
  Contains BOTH 'สิทธิบัตร' (9 chars) and 'สิทธิบัตรการประดิษฐ์' (20 chars)

  Correct order (long→short): ['การละเมิด', 'สิทธิบัตรการประดิษฐ์', 'โดยจำเลย']
  Wrong order   (short→long): ['การละเมิดสิทธิ', 'บัตรการประดิษฐ์โดยจำเลย']
"""
