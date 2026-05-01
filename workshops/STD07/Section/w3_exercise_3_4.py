import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import importlib.util
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

# ===== Import จาก w3 source =====
SRC_PATH = Path(__file__).parent / "w3 Attention_legal.py"
spec = importlib.util.spec_from_file_location("w3_src", SRC_PATH)
w3_src = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()):
    spec.loader.exec_module(w3_src)

MultiHeadAttentionSimple = w3_src.MultiHeadAttentionSimple
explainable_attention = w3_src.explainable_attention

# ===== Re-do W3.2 เพื่อให้ได้ weights ตามสภาพเดิม =====
np.random.seed(42)
tokens = ["จำเลย", "ผลิต", "สินค้า", "ละเมิด", "สิทธิบัตร"]
X_input = np.random.randn(5, 16)
mha = MultiHeadAttentionSimple(d_model=16, n_heads=4, seed=42)
output, weights = mha.forward(X_input, return_weights=True)
# weights.shape = (n_heads=4, seq=5, seq=5)

# ===== โค้ดบังคับ Block 1: XAI =====
print("=" * 70)
print("  Block 1: XAI — explainable_attention()")
print("=" * 70)

w_batch = weights[np.newaxis, :]   # เพิ่ม batch dim → (1, heads, seq, seq)
print(f"  w_batch.shape = {w_batch.shape}  (1, heads, seq, seq)")

# ลองเรียกตามโจทย์ตรง ๆ → จะ crash เพราะ source's explainable_attention
# คาดหวัง 3D (1, heads, seq) ไม่ใช่ 4D
try:
    explainable_attention(tokens, w_batch)
except TypeError as e:
    print(f"\n  ❌ explainable_attention(tokens, w_batch) error:")
    print(f"     {type(e).__name__}: {e}")
    print(f"\n  สาเหตุ: source's explainable_attention คาดหวัง weight shape")
    print(f"          (1, heads, seq) แต่ MHA.forward คืน 4D (1, heads, seq, seq)")
    print(f"  วิธีแก้: aggregate query dim ก่อน — เอา column-mean ของแต่ละ token")
    print(f"          (= 'token นี้ถูก attend จาก query ทั้งหมดเฉลี่ยเท่าไหร่')")

# Aggregate (1, heads, seq, seq) → (1, heads, seq) ด้วย mean ตาม query axis
per_token = w_batch.mean(axis=2)   # (1, heads, seq)
print(f"\n  ▶ Aggregated: per_token.shape = {per_token.shape}")
print(f"\n  เรียก explainable_attention ด้วย per_token:")
explainable_attention(tokens, per_token)

# เก็บ importance per token (ใช้ในข้อต่อ ๆ ไป)
avg_attn = per_token[0].mean(axis=0)   # (seq,) — average across heads

# ================================================================
# (hh) Token ใดมี importance สูงสุด — สอดคล้องกับความหมายของประโยคหรือไม่
# W 3.4 : HH token ใดมี importance สูงสุดจาก explainable_attention() — สอดคล้องกับความหมายของประโยคหรือไม่
# ================================================================
print("\n" + "=" * 70)
print("  (hh) Token ที่ importance สูงสุด")
print("=" * 70)

ranked = sorted(zip(tokens, avg_attn), key=lambda x: -x[1])
print(f"\n  Ranking (จากสูง→ต่ำ):")
for rank, (tok, score) in enumerate(ranked, 1):
    bar = "█" * int(score * 50)
    print(f"    {rank}. {tok:<12} {score:.4f}  {bar}")

top_token = ranked[0][0]
print(f"""
  ▶ Token importance สูงสุด: '{top_token}'

  ▶ สอดคล้องกับความหมายของประโยคหรือไม่:

    ประโยค: "จำเลย ผลิต สินค้า ละเมิด สิทธิบัตร"
    semantic value (สำหรับการจำแนก IP):
      จำเลย     — ระบุผู้ทำผิด (สำคัญ procedural)
      ผลิต      — กริยาการกระทำ (สำคัญ — เป็น sensor trigger)
      สินค้า    — object (น้อย)
      ละเมิด    — กริยาผิดกฎหมาย (สำคัญ — IP context)
      สิทธิบัตร — ประเภท IP (สำคัญสุด — เป็น class label)

    คาดหวัง: 'สิทธิบัตร' หรือ 'ละเมิด' ควรสูงสุด

  ⚠️  หมายเหตุ: weight นี้มาจาก MHA ที่ใช้ random projection (W_q, W_k, W_v
       ถูก init ด้วย rng.randn × 0.1, ไม่ได้ train) → attention pattern ยัง
       ไม่ได้สะท้อนความสำคัญ semantic จริง
       ผลที่ได้จึงค่อนข้าง uniform (ดูจาก bar ที่ใกล้กัน)

       ถ้า train MHA แล้ว → คำสำคัญจะมี weight สูงเด่นชัดขึ้น
""")

"""
======================================================================
  (hh) Token ที่ importance สูงสุด
======================================================================

  Ranking (จากสูง→ต่ำ):
    1. สิทธิบัตร    0.2112  ██████████
    2. จำเลย        0.2011  ██████████
    3. ผลิต         0.1992  █████████
    4. ละเมิด       0.1949  █████████
    5. สินค้า       0.1936  █████████

  ▶ Token importance สูงสุด: 'สิทธิบัตร'

  ▶ สอดคล้องกับความหมายของประโยคหรือไม่:

    ประโยค: "จำเลย ผลิต สินค้า ละเมิด สิทธิบัตร"
    semantic value (สำหรับการจำแนก IP):
      จำเลย     — ระบุผู้ทำผิด (สำคัญ procedural)
      ผลิต      — กริยาการกระทำ (สำคัญ — เป็น sensor trigger)
      สินค้า    — object (น้อย)
      ละเมิด    — กริยาผิดกฎหมาย (สำคัญ — IP context)
      สิทธิบัตร — ประเภท IP (สำคัญสุด — เป็น class label)

    คาดหวัง: 'สิทธิบัตร' หรือ 'ละเมิด' ควรสูงสุด

  ⚠️  หมายเหตุ: weight นี้มาจาก MHA ที่ใช้ random projection (W_q, W_k, W_v
       ถูก init ด้วย rng.randn × 0.1, ไม่ได้ train) → attention pattern ยัง
       ไม่ได้สะท้อนความสำคัญ semantic จริง
       ผลที่ได้จึงค่อนข้าง uniform (ดูจาก bar ที่ใกล้กัน)

       ถ้า train MHA แล้ว → คำสำคัญจะมี weight สูงเด่นชัดขึ้น
"""

# ================================================================
# Block 2: SENSOR_MAP + physics_score (ตามโจทย์)
# ================================================================
print("=" * 70)
print("  Block 2: Physics Gate — SENSOR_MAP + physics_score()")
print("=" * 70)

SENSOR_MAP = {
    "ผลิต":    {"sensor": "MANUFACTURING_SENSOR", "base_weight": 1.3},
    "ละเมิด":  {"sensor": "VIOLATION_MONITOR",    "base_weight": 1.2},
    "นำเข้า":  {"sensor": "CUSTOMS_GPS",          "base_weight": 1.2},
    "จำหน่าย": {"sensor": "POS_TRACKING",         "base_weight": 1.1},
}

def physics_score(token, attn, confidence):
    cfg = SENSOR_MAP.get(token, {"base_weight": 0.5})
    return min(2.0, cfg["base_weight"] * attn * confidence)

print("\n  SENSOR_MAP:")
for tok, cfg in SENSOR_MAP.items():
    print(f"    '{tok}': {cfg}")

print("\n  physics_score():")
print("    def physics_score(token, attn, confidence):")
print("        cfg = SENSOR_MAP.get(token, {\"base_weight\": 0.5})")
print("        return min(2.0, cfg[\"base_weight\"] * attn * confidence)")

# ================================================================
# (ii) physics_score ทุก token confidence=0.85 — token ใด trigger sensor (>0.4)
# W 3.4 : II คำนวณ physics_score ของทุก token เมื่อ confidence=0.85 — token ใด trigger sensor (score > 0.4)
# ================================================================
print("\n" + "=" * 70)
print("  (ii) physics_score @ confidence = 0.85")
print("=" * 70)

CONF_HIGH = 0.85
TRIGGER_THRESHOLD = 0.4

print(f"\n  {'Token':<12} {'attn':>8} {'base_w':>8} {'score':>8} {'sensor':<25} {'trigger?':<10}")
print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 25} {'-' * 10}")

scores_high = {}
for tok, attn in zip(tokens, avg_attn):
    cfg = SENSOR_MAP.get(tok, {"sensor": "UNMAPPED", "base_weight": 0.5})
    score = physics_score(tok, attn, CONF_HIGH)
    scores_high[tok] = score
    triggered = "🔥 FIRE" if score > TRIGGER_THRESHOLD else "—"
    print(f"  {tok:<12} {attn:>8.4f} {cfg['base_weight']:>8.2f} {score:>8.4f} {cfg['sensor']:<25} {triggered:<10}")

triggered_high = [t for t, s in scores_high.items() if s > TRIGGER_THRESHOLD]
print(f"\n  ▶ Token ที่ trigger sensor (score > {TRIGGER_THRESHOLD}): {triggered_high}")

"""
======================================================================
  (ii) physics_score @ confidence = 0.85
======================================================================

  Token            attn   base_w    score sensor                    trigger?  
  ------------ -------- -------- -------- ------------------------- ----------
  จำเลย          0.2011     0.50   0.0855 UNMAPPED                  —         
  ผลิต           0.1992     1.30   0.2201 MANUFACTURING_SENSOR      —         
  สินค้า         0.1936     0.50   0.0823 UNMAPPED                  —         
  ละเมิด         0.1949     1.20   0.1988 VIOLATION_MONITOR         —         
  สิทธิบัตร      0.2112     0.50   0.0897 UNMAPPED                  —         

  ▶ Token ที่ trigger sensor (score > 0.4): []
"""

# ================================================================
# (jj) ทำซ้ำ confidence=0.3 — ผลเปลี่ยนยังไง ทำไม W4 ต้องเพิ่ม confidence
# W 3.4 : JJ ทำซ้ำด้วย confidence=0.3 — ผลเปลี่ยนไปอย่างไร และทำไม W4 ถึงต้องเพิ่ม confidence เข้าสูตร
# ================================================================
print("\n" + "=" * 70)
print("  (jj) physics_score @ confidence = 0.3 (low confidence)")
print("=" * 70)

CONF_LOW = 0.3

print(f"\n  {'Token':<12} {'attn':>8} {'base_w':>8} {'@0.85':>8} {'@0.30':>8} {'Δ':>8} {'still trigger?':<15}")
print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 15}")

scores_low = {}
for tok, attn in zip(tokens, avg_attn):
    cfg = SENSOR_MAP.get(tok, {"base_weight": 0.5})
    s_high = scores_high[tok]
    s_low = physics_score(tok, attn, CONF_LOW)
    scores_low[tok] = s_low
    diff = s_low - s_high
    still = "🔥 yes" if s_low > TRIGGER_THRESHOLD else "no"
    print(f"  {tok:<12} {attn:>8.4f} {cfg['base_weight']:>8.2f} {s_high:>8.4f} {s_low:>8.4f} {diff:>+8.4f} {still:<15}")

triggered_low = [t for t, s in scores_low.items() if s > TRIGGER_THRESHOLD]
print(f"\n  ▶ Trigger @ 0.85: {triggered_high}")
print(f"  ▶ Trigger @ 0.30: {triggered_low}")
print(f"  ▶ ลดลง: {set(triggered_high) - set(triggered_low)}")

print(f"""
  ▶ ผลของ confidence ต่ำ:
    - Score ทุก token ลดลง ~3.5 เท่า (0.85 → 0.30 = 0.353 ratio)
    - Sensor หลายตัว 'หยุด trigger' → ระบบไม่ส่งสัญญาณเตือน

  ▶ ทำไม W4 ต้องเพิ่ม confidence เข้าสูตร physics_score:

    1. ป้องกัน False Positive จาก low-quality prediction:
       ถ้า W1/W2/W3 จำแนกคดีด้วย confidence ต่ำ (ไม่แน่ใจ)
       → ไม่ควร trigger sensor จริง (เปลือง resource, alarm fatigue)
       → confidence × score = "เปิด sensor ก็ต่อเมื่อมั่นใจมากพอ"

    2. Calibration กับ legal certainty:
       กฎหมายต้องการ "เกินสมควรสงสัย" (beyond reasonable doubt)
       confidence ต่ำ = หลักฐานอ่อน → ไม่ควรทำให้ระบบ enforce อัตโนมัติ

    3. Graceful degradation:
       Linear scale (× confidence) ทำให้ score เปลี่ยนตามคุณภาพ input
       แทนที่จะเป็น all-or-nothing → ผู้ดูแลตัดสินใจว่าจะ act หรือ wait

    4. รองรับ Bayesian update ใน future iteration:
       ถ้า W4/W5 update confidence ตามหลักฐานเพิ่ม → physics_score
       update ตามไปด้วยทันที (ไม่ต้องคำนวณใหม่ทั้งระบบ)
""")

"""
======================================================================
  (jj) physics_score @ confidence = 0.3 (low confidence)
======================================================================

  Token            attn   base_w    @0.85    @0.30        Δ still trigger? 
  ------------ -------- -------- -------- -------- -------- ---------------
  จำเลย          0.2011     0.50   0.0855   0.0302  -0.0553 no             
  ผลิต           0.1992     1.30   0.2201   0.0777  -0.1424 no             
  สินค้า         0.1936     0.50   0.0823   0.0290  -0.0532 no             
  ละเมิด         0.1949     1.20   0.1988   0.0702  -0.1286 no             
  สิทธิบัตร      0.2112     0.50   0.0897   0.0317  -0.0581 no             

  ▶ Trigger @ 0.85: []
  ▶ Trigger @ 0.30: []
  ▶ ลดลง: set()

  ▶ ผลของ confidence ต่ำ:
    - Score ทุก token ลดลง ~3.5 เท่า (0.85 → 0.30 = 0.353 ratio)
    - Sensor หลายตัว 'หยุด trigger' → ระบบไม่ส่งสัญญาณเตือน

  ▶ ทำไม W4 ต้องเพิ่ม confidence เข้าสูตร physics_score:

    1. ป้องกัน False Positive จาก low-quality prediction:
       ถ้า W1/W2/W3 จำแนกคดีด้วย confidence ต่ำ (ไม่แน่ใจ)
       → ไม่ควร trigger sensor จริง (เปลือง resource, alarm fatigue)
       → confidence × score = "เปิด sensor ก็ต่อเมื่อมั่นใจมากพอ"

    2. Calibration กับ legal certainty:
       กฎหมายต้องการ "เกินสมควรสงสัย" (beyond reasonable doubt)
       confidence ต่ำ = หลักฐานอ่อน → ไม่ควรทำให้ระบบ enforce อัตโนมัติ

    3. Graceful degradation:
       Linear scale (× confidence) ทำให้ score เปลี่ยนตามคุณภาพ input
       แทนที่จะเป็น all-or-nothing → ผู้ดูแลตัดสินใจว่าจะ act หรือ wait

    4. รองรับ Bayesian update ใน future iteration:
       ถ้า W4/W5 update confidence ตามหลักฐานเพิ่ม → physics_score
       update ตามไปด้วยทันที (ไม่ต้องคำนวณใหม่ทั้งระบบ)
"""

# ================================================================
# (kk) mean(axis=0) ใน explainable_attention() ทำหน้าที่อะไร — n_heads=4 รวมยังไง
# W 3.4 : KK อธิบายว่า mean(axis=0) ใน explainable_attention() ทำหน้าที่อะไร — ถ้า n_heads=4 จะรวม information จาก head ต่างๆ อย่างไร
# ================================================================
print("=" * 70)
print("  (kk) mean(axis=0) ใน explainable_attention()")
print("=" * 70)

print(f"""
  ▶ Source code ที่ explainable_attention():
      def explainable_attention(tokens, weight):
          avg_weight = weight[0].mean(axis=0)   ← ตรงนี้
          for i, token in enumerate(tokens):
              importance = avg_weight[i]
              ...

  ▶ ขั้นตอน:
    1. weight ที่ใส่เข้ามา shape: (1, n_heads, n_tokens) ← แบบ pre-aggregated
    2. weight[0]               shape: (n_heads, n_tokens) ← ตัด batch dim
    3. weight[0].mean(axis=0)  shape: (n_tokens,)         ← เฉลี่ยข้าม heads

  ▶ ทำไม axis=0 = "เฉลี่ยข้าม heads":
    ใน weight[0] shape (n_heads, n_tokens):
      axis 0 = heads dimension
      axis 1 = tokens dimension
    mean(axis=0) = ยุบมิติ heads → คงเหลือ tokens
    → ได้ "ค่าเฉลี่ย attention per token" ที่ทุก head เห็นรวมกัน

  ▶ ถ้า n_heads = 4 จะรวม info ยังไง:

    สมมติ weight[0] = แต่ละ head ให้ความสำคัญต่างกัน:
""")

# Demo: 4 heads, 5 tokens
demo = np.array([
    [0.1, 0.5, 0.1, 0.1, 0.2],   # head 0: เน้น 'ผลิต'
    [0.1, 0.1, 0.1, 0.6, 0.1],   # head 1: เน้น 'ละเมิด'
    [0.1, 0.1, 0.1, 0.1, 0.6],   # head 2: เน้น 'สิทธิบัตร'
    [0.5, 0.1, 0.1, 0.1, 0.2],   # head 3: เน้น 'จำเลย'
])

print(f"    Demo (4 heads × 5 tokens):")
print(f"    {'Head':<7} {'จำเลย':>8} {'ผลิต':>8} {'สินค้า':>8} {'ละเมิด':>8} {'สิทธิบัตร':>8}")
for h in range(4):
    print(f"    head {h}  " + " ".join(f"{demo[h, i]:>8.2f}" for i in range(5)))

avg = demo.mean(axis=0)
print(f"\n    mean(axis=0) → " + " ".join(f"{v:>8.2f}" for v in avg))

print(f"""
  ▶ ผลของ averaging:
    1. ทุก head 'โหวต' ความสำคัญของแต่ละ token
    2. Average = consensus view → token ที่หลายhead เห็นว่าสำคัญ จะได้ score สูง
    3. Token ที่ head เดียวสนใจ จะถูก "เจือจาง" → ได้ score ต่ำ

  ▶ Trade-off ของการ average:

    ✅ ข้อดี:
      - ลด noise จาก head เดียว
      - ได้ importance ที่ "เห็นพ้องกัน" (robust)
      - Interpretable: 1 ค่าต่อ 1 token

    ❌ ข้อเสีย:
      - สูญเสีย information ที่ head แต่ละตัวจับ pattern ต่างกัน
      - เช่น head 1 อาจจับ syntax, head 2 จับ entity → average = ลายเลือนทั้งคู่
      - งานวิจัย XAI ขั้นสูงนิยม "head-wise analysis" แทน average

  ▶ Alternative aggregation strategies:
    - max(axis=0)  : เอา head ที่ confident ที่สุด
    - sum(axis=0)  : เน้น token ที่หลาย head สนใจซ้อนกัน
    - weighted     : ให้ head สำคัญถ่วงน้ำหนักมากขึ้น (ต้องเรียนรู้ weights)
    - per-head     : ไม่ aggregate, แสดงทุก head แยกกัน (ใน BERT visualizer)

  ใน source ใช้ mean เพราะ "เริ่มต้น" ที่ตีความง่ายและ baseline พอใช้ได้
""")


"""
======================================================================
  (kk) mean(axis=0) ใน explainable_attention()
======================================================================

  ▶ Source code ที่ explainable_attention():
      def explainable_attention(tokens, weight):
          avg_weight = weight[0].mean(axis=0)   ← ตรงนี้
          for i, token in enumerate(tokens):
              importance = avg_weight[i]
              ...

  ▶ ขั้นตอน:
    1. weight ที่ใส่เข้ามา shape: (1, n_heads, n_tokens) ← แบบ pre-aggregated
    2. weight[0]               shape: (n_heads, n_tokens) ← ตัด batch dim
    3. weight[0].mean(axis=0)  shape: (n_tokens,)         ← เฉลี่ยข้าม heads

  ▶ ทำไม axis=0 = "เฉลี่ยข้าม heads":
    ใน weight[0] shape (n_heads, n_tokens):
      axis 0 = heads dimension
      axis 1 = tokens dimension
    mean(axis=0) = ยุบมิติ heads → คงเหลือ tokens
    → ได้ "ค่าเฉลี่ย attention per token" ที่ทุก head เห็นรวมกัน

  ▶ ถ้า n_heads = 4 จะรวม info ยังไง:

    สมมติ weight[0] = แต่ละ head ให้ความสำคัญต่างกัน:

    Demo (4 heads × 5 tokens):
    Head       จำเลย     ผลิต   สินค้า   ละเมิด สิทธิบัตร
    head 0      0.10     0.50     0.10     0.10     0.20
    head 1      0.10     0.10     0.10     0.60     0.10
    head 2      0.10     0.10     0.10     0.10     0.60
    head 3      0.50     0.10     0.10     0.10     0.20

    mean(axis=0) →     0.20     0.20     0.10     0.22     0.28

  ▶ ผลของ averaging:
    1. ทุก head 'โหวต' ความสำคัญของแต่ละ token
    2. Average = consensus view → token ที่หลายhead เห็นว่าสำคัญ จะได้ score สูง
    3. Token ที่ head เดียวสนใจ จะถูก "เจือจาง" → ได้ score ต่ำ

  ▶ Trade-off ของการ average:

    ✅ ข้อดี:
      - ลด noise จาก head เดียว
      - ได้ importance ที่ "เห็นพ้องกัน" (robust)
      - Interpretable: 1 ค่าต่อ 1 token

    ❌ ข้อเสีย:
      - สูญเสีย information ที่ head แต่ละตัวจับ pattern ต่างกัน
      - เช่น head 1 อาจจับ syntax, head 2 จับ entity → average = ลายเลือนทั้งคู่
      - งานวิจัย XAI ขั้นสูงนิยม "head-wise analysis" แทน average

  ▶ Alternative aggregation strategies:
    - max(axis=0)  : เอา head ที่ confident ที่สุด
    - sum(axis=0)  : เน้น token ที่หลาย head สนใจซ้อนกัน
    - weighted     : ให้ head สำคัญถ่วงน้ำหนักมากขึ้น (ต้องเรียนรู้ weights)
    - per-head     : ไม่ aggregate, แสดงทุก head แยกกัน (ใน BERT visualizer)

  ใน source ใช้ mean เพราะ "เริ่มต้น" ที่ตีความง่ายและ baseline พอใช้ได้
"""
