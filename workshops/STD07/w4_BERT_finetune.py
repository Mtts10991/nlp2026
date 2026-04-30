# Model Selection & Drvice Detection
# BERT ถูกเทรน (Pre-Train) MLM อาศัยการเรียนรู้บริบทของคำรอบๆ -> การทำ Fine-tune คือการนำ ค่า Weight ที่เรียนมาแล้ว มาปรับจูนให้เข้ากับงานที่เราต้องการ
# จากนั้นนำมาต่อยอดทำนายผลที่เราต้องการต่อไป
# เลือก Model ตามภาษา และขนาดของข้อมูล
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

MODEL_REGISTRY = {
    "mbert": {
        "hf_name": "bert-base-multilingual-cased",
        "paramiters": "110M",
        "thai_coverage": "1%",
        "decption": "ความสามารถในการเข้าใจภาษาไทยน้อยมาก",
        "device" : "cpu"
    },
    "xlmr": {
        "hf_name": "xlm-roberta-base",
        "paramiters": "270M",
        "thai_coverage": "5%",
        "decption": "ดีกว่า M-BERT เพราะคือ RoBERTa ที่เทรนบนข้อมูลที่หลากหลายกว่า",
        "device" : "cpu"
    },
    "wangchanberta": {
        "hf_name": "airesearch/wangchanberta-base-att-spm-uncased",
        "paramiters": "110M",
        "thai_coverage": "100%",
        "decption": "ดีที่สุดสำหรับภาษาไทย เพราะคือ wangchanberta ที่เทรนบนข้อมูลภาษาไทยเท่านั้น",
        "device" : "cpu"
    }
}

def show_model_comparison():
    print(" " * 70)
    print("=" * 70)
    print(f"{'MODEL':<16} | {'PARAMETERS':<15} | {'THAI COVERAGE':<15} | {'DESCRIPTION'}")
    print("-" * 70)
    for model, info in MODEL_REGISTRY.items():
        print(f"{model:<16} | {info['paramiters']:<15} | {info['thai_coverage']:<15} | {info['decption']}")
    print("=" * 70)
    print(" " * 70)

show_model_comparison()

# Tokenization (BERT ใช้ Subword Tokenization แก้ปัญหา OOV - Out Of Vocabulary)
# เปรียบเทียบ 3 แบบ
class MockTokenizer:
    # จำลอง BERT Tokenizer เพื่อแสดงการทำงาน
    LEGAL_TERMS = ["ภูมิปัญญาท้องถิ่น", "สิทธิการประดิษฐ์", "อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ" ]
    def endcode(self, texts, max_lenght=24):
        """แปลงข้อความ ->  Input_ids, Attention_mask, Token_type_id
            1. Input_ids: [CLS] + Token1 + Token2 + ... + TokenN + [SEP]
            2. Attention_mask: [1, 1, 1, ..., 1, 1] (ความยาวเท่ากับ Input_ids) -> 1=real token, 0=pad token
            3. Token_type_id: [0, 0, 0, ..., 0, 0] (ความยาวเท่ากับ Input_ids) 
                (สำหรับ single sentence)
        """
        input_ids, attention_mask, token_type_ids = [], [], []
        for text in texts:
            # [CLS: 1], [SEP: 2]
            ids = [1]+[ord(c) % 5000+100 for c in text[:max_lenght-2]]+[2]
            # padding & attention mask
            att_mask = [1]*len(ids)
            padding_len = max_lenght - len(ids)
            # mask คือ[1,1,1,1,0,0,0,0] - 1 คือ นับมา, 0 คือ ไม่นับมา (Padding ไม่นับมาด้วย)
            mask = [1] * len(ids) + [0] * padding_len
            ids = ids + [0]*padding_len
            att_mask = att_mask+[0]*padding_len
            token_type_ids.append([0]*max_lenght)

            input_ids.append(ids)
            attention_mask.append(att_mask)
            
        return {
            "input_ids" : np.array(input_ids, dtype=np.int32),
            "attention_mask" : np.array(attention_mask, dtype=np.int32),
            "token_type_ids" : np.array(token_type_ids, dtype=np.int32),
            "mask" : np.array(mask, dtype=np.int32)
        }
    
    def show(self, text, max_lenght=32):
        endcodeshow = self.endcode([text], max_lenght)
        input_ids = endcodeshow['input_ids'][0]
        att_mask = endcodeshow['attention_mask'][0]
        tokens = ["[CLS]"] + [text[:max_lenght-2]] + ["[SEP]"]
        real_len = att_mask.sum()
        print(f"\nText : {text}")
        print(f"input_ids (BERT vocab) : {input_ids[:real_len].tolist()} + [PADx{max_lenght - real_len}]={len(input_ids)}")
        print(f"input_ids (BERT vocab) : {input_ids.tolist()}")
        print(f"att_mask : {att_mask[:real_len].tolist()} + [PADx{max_lenght - real_len}]={len(att_mask)}")
        print(f"mask : {att_mask.tolist()}")
        print(f"[CLS] = 1, [SEP] = 2, PAD = 0 (ความหมาย) เติมให้ครบ : {max_lenght} ตัว")

    def show2(self, text, max_length=32): 
        enc = self.endcode([text], max_length) 
        ids = enc["input_ids"][0] 
        mask = enc["attention_mask"][0] 
        real_len = mask.sum() 
        print(f"\n ข้อความ : '{text}'") 
        print(f" input_ids : {ids.tolist()}") # ← แสดงทั้ง array รวม 0 
        print(f" mask : {mask.tolist()}") 
        print(f" real tokens: {real_len} PAD: {max_length - real_len}") 
        print(f"\n แยกส่วน:") 
        print(f" [CLS] = {ids[0]}") 
        print(f" tokens= {ids[1:real_len-1].tolist()}")
        print(f" [SEP] = {ids[real_len-1]}") 
        print(f" [PAD] = {ids[real_len:].tolist()}") # ← แสดง 0s
        
    def decode(self, input_ids):
        """
        
        """
tok = MockTokenizer()
text= "ผู้ต้องหาละเมิดสิทธิบัตร"
tok.show2(text, max_length=32)
# print(tok.endcode([text], max_lenght=16))
print("=" * 70)
for term in tok.LEGAL_TERMS:
    print(f" + {term}")
print("=" * 70)
print(" " * 70)
# ── Vocabulary Expansion Demo ────────────────────────────────── 
def vocab_expansion_demo():
    print("=" * 70)
    print(" STEP 1: ก่อน expansion — คำใหม่ถูกตัดเป็นชิ้น")
    print("=" * 70) 
    tok.show2("ทรัพย์สินทางปัญญา")
# print("=" * 70) tok.show2("สิทธิบัตรการประดิษฐ์")
print("\n" + "=" * 70)
print(" STEP 2: เพิ่มคำใหม่เข้า vocab (Vocabulary Expansion)")
print("=" * 70) # vocab ปกติ + เพิ่มคำกฎหมาย
base_vocab_size = 5000
new_vocab = {term:base_vocab_size + i
             for i, term in enumerate(MockTokenizer.LEGAL_TERMS)}
print(f"\n vocab เดิม : {base_vocab_size} คำ")
print(f" เพิ่มคำใหม่: {len(new_vocab)} คำ")
print(f" vocab ใหม่ : {base_vocab_size +len(new_vocab)} คำ\n")
for term, idx in new_vocab.items():
    print(f" '{term}' → id {idx} (token ใหม่ weight = random ❗)")
    print("\n" + "=" * 70)
    print(" STEP 3: ทำไมต้อง Warm-up ก่อน Fine-tune")
    print("=" * 70)
    print(""" ปัญหา: คำเดิม → weights ผ่าน pre-train มาแล้ว (มีความหมาย) คำใหม่ → weights = random ❗ (ยังไม่มีความหมาย) ถ้า fine-tune ทุก layer พร้อมกันเลย: gradient จากคำใหม่ (random) จะรบกวน weights เดิม → โมเดลลืมสิ่งที่เรียนมา = Catastrophic Forgetting ❌ วิธีแก้ — Warm-up 3 ขั้นตอน: ขั้น 1 │ Freeze ทุก layer ยกเว้น embedding │ train แค่ embedding 2-3 epochs│ → คำใหม่เริ่มมีความหมาย ขั้น 2 │ Unfreeze ทุก layer │ train ด้วย LR ต่ำ (2e-5) │ → ปรับ weights ทั้งหมดพร้อมกันขั้น 3 │ Fine-tune จนกว่า val_loss นิ่ง │ → โมเดลพร้อมใช้งาน ✅ """) 
    print("=" * 70) 
    print(" STEP 4: จำลอง embedding weight ก่อน/หลัง warm-up")
    print("=" * 70)
    np.random.seed(42)
    d_model = 8 # embedding ของคำเดิม (pretrained) — มีค่าชัดเจน
    old_emb = np.array([0.82, -0.34, 0.56, 0.91, -0.12, 0.67, -0.45, 0.23]) # embedding ของคำใหม่ก่อน warm-up — random
    new_before = np.random.randn(d_model) * 0.02 # หลัง warm-up — เริ่มมีทิศทาง (จำลอง) 
    new_after = old_emb *0.6 + np.random.randn(d_model) * 0.1 
    print(f"\n คำเดิม 'ละเมิด' : {old_emb.round(2)}")
    print(f" คำใหม่ ก่อน warm-up : {new_before.round(2)} ← random")
    print(f" คำใหม่ หลัง warm-up : {new_after.round(2)} ← มีทิศทางแล้ว")
    # cosine similarity
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\n cosine similarity กับ 'ละเมิด':")
    print(f" ก่อน warm-up : {cosine(old_emb, new_before):.3f} (ไม่เกี่ยวกัน)")
    print(f" หลัง warm-up : {cosine(old_emb, new_after):.3f} (ใกล้เคียงกัน)")
    # ── รัน ──────────────────────────────────────────────────────

print()
vocab_expansion_demo()
print("=" * 70)
print(" " * 70)

# BERT Endcoder 

# # Masked Language Model (MLM)
# # BERT ถูกเทรน MLM อาศัยการเรียนรู้บริบทของคำรอบๆ -> การทำ Fine-tune คือการนำ ค่า Weight ที่เรียนมาแล้ว มาปรับจูนให้เข้ากับงานที่เราต้องการ
# # จากนั้นนำมาต่อยอดทำนายผลที่เราต้องการต่อไป
# # เลือก Model ตามภาษา และขนาดของข้อมูล
# from transformers import AutoTokenizer
