# =============================================================================
# Sinusoidal Position Encoding (การเข้ารหัสตำแหน่งด้วยฟังก์ชัน sin/cos)
# =============================================================================
# ทำไมต้องมี Position Encoding?
# -----------------------------------------------------------------------------
# โมเดล Transformer (เช่น BERT, GPT) ใช้ self-attention ซึ่ง "ไม่มีลำดับ"
# โดยธรรมชาติ — ถ้าสลับคำในประโยค ผลลัพธ์ก็ยังเหมือนเดิม
# ตัวอย่าง: "หมา กัด คน" กับ "คน กัด หมา" จะถูกมองเหมือนกัน ❌
#
# เราจึงต้อง "ฉีด" ข้อมูลตำแหน่ง (position) เข้าไปใน embedding ของแต่ละคำ
# โดย Vaswani et al. (2017) เสนอให้ใช้ฟังก์ชัน sin และ cos ที่มีความถี่ต่างกัน
# ในแต่ละ dimension เพราะ:
#   1) ค่าอยู่ในช่วง [-1, 1] เสมอ (ไม่ทำให้ embedding ผิดเพี้ยน)
#   2) แต่ละตำแหน่งมี "ลายเซ็น" (signature) ที่ไม่ซ้ำกัน
#   3) โมเดลเรียนรู้ระยะห่างสัมพัทธ์ได้ง่าย (PE_pos+k เขียนเป็น linear ของ PE_pos)
#
# สูตร:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# =============================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SinusoidalPositionEncoding = คลาสที่สร้าง position encoding ตามสูตรของ Vaswani et al. (2017)
class SinusoidalPositionEncoding:
    def __init__(self, d_model=16, max_len=10):
        # d_model = ขนาดของ embedding vector ของแต่ละคำ (ในที่นี้ใช้ 16 เพื่อให้ดูง่าย)
        # max_len = ความยาวสูงสุดของ sequence ที่รองรับ (จำนวนคำสูงสุดในประโยค)
        self.d_model = d_model
        self.max_len = max_len
        # สร้างตาราง position encoding ขนาด (1, max_len, d_model) ไว้ล่วงหน้า
        # shape (1, ...) เพื่อให้ broadcast กับ batch ได้สะดวกตอน forward
        self.pe = self._create_position_encoding()

    def _create_position_encoding(self):
        # ----- สร้าง matrix ว่างขนาด (max_len, d_model) -----
        # แต่ละแถว = 1 ตำแหน่ง, แต่ละคอลัมน์ = 1 dimension ของ embedding
        pe = torch.zeros(self.max_len, self.d_model)

        # ----- สร้าง vector ตำแหน่ง [0, 1, 2, ..., max_len-1] -----
        # unsqueeze(1) ทำให้ shape เป็น (max_len, 1) เพื่อ broadcast กับ div_term
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)

        # ----- คำนวณ "ความถี่" (frequency) สำหรับแต่ละคู่ของ dimension -----
        # ใช้ exp(log) แทน power โดยตรงเพื่อความเสถียรเชิงตัวเลข (numerical stability)
        # div_term[i] = 1 / 10000^(2i/d_model)
        # ยิ่ง dimension สูง ความถี่ยิ่งต่ำ (คลื่นยาวขึ้น) → จับ pattern ระยะไกลได้
        # ยิ่ง dimension ต่ำ ความถี่ยิ่งสูง (คลื่นสั้น) → จับ pattern ระยะใกล้ได้
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )

        # ----- ใส่ค่า sin ใน dimension เลขคู่ (0, 2, 4, ...) -----
        pe[:, 0::2] = torch.sin(position * div_term)
        # ----- ใส่ค่า cos ใน dimension เลขคี่ (1, 3, 5, ...) -----
        pe[:, 1::2] = torch.cos(position * div_term)

        # เพิ่ม batch dimension → shape สุดท้าย: (1, max_len, d_model)
        return pe.unsqueeze(0)

    # -------------------------------------------------------------------------
    # แสดงผลแบบ tensor (default ของ PyTorch) — เหมาะกับการ debug
    # -------------------------------------------------------------------------
    def show_position_encoding(self, seq_len=10):
        print("-" * 70)
        print(f"Position Encoding for sequence length {seq_len} tokens (shape: {self.pe.shape}):")
        print("-" * 70)
        for position in range(seq_len):
            print(f"Position :: {position}: {self.pe[0, position, :]}")
        return self.pe

    # -------------------------------------------------------------------------
    # แสดงผลแบบ DataFrame — อ่านง่าย เห็นเป็นตารางชัดเจน
    # -------------------------------------------------------------------------
    def show_as_dataframe(self, seq_len=10, decimals=4):
        # ดึงข้อมูลจาก tensor (1, max_len, d_model) → numpy 2D (seq_len, d_model)
        pe_2d = self.pe[0, :seq_len, :].numpy()

        # สร้าง DataFrame โดย:
        #   - แถว (index)   = ตำแหน่งของคำในประโยค (pos_0, pos_1, ...)
        #   - คอลัมน์       = dimension ของ embedding (dim_0, dim_1, ...)
        #     โดย dim เลขคู่ = sin, dim เลขคี่ = cos
        df = pd.DataFrame(
            pe_2d,
            index=[f"pos_{i}" for i in range(seq_len)],
            columns=[
                f"dim_{j} ({'sin' if j % 2 == 0 else 'cos'})"
                for j in range(self.d_model)
            ],
        )
        print("\n=== Position Encoding (DataFrame view) ===")
        print(df.round(decimals))
        return df

    # -------------------------------------------------------------------------
    # แสดงผลแบบ Heatmap — เห็น "ลวดลายคลื่น" ของ sin/cos ได้ชัดเจน
    # -------------------------------------------------------------------------
    def plot_heatmap(self, seq_len=10):
        pe_2d = self.pe[0, :seq_len, :].numpy()

        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ----- ภาพที่ 1: Heatmap ของค่าทั้งหมด -----
        # แกน X = dimension (0 ถึง d_model-1)
        # แกน Y = ตำแหน่ง (0 ถึง seq_len-1)
        # สี    = ค่าของ PE ในช่วง [-1, 1] (แดง=บวก, น้ำเงิน=ลบ)
        im = axes[0].imshow(pe_2d, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        axes[0].set_xlabel("Dimension (มิติของ embedding)")
        axes[0].set_ylabel("Position (ตำแหน่งคำในประโยค)")
        axes[0].set_title("Position Encoding Heatmap\n(แดง=บวก, น้ำเงิน=ลบ)")
        axes[0].set_xticks(range(self.d_model))
        axes[0].set_yticks(range(seq_len))
        plt.colorbar(im, ax=axes[0])

        # ----- ภาพที่ 2: เส้นกราฟแสดง "คลื่น" ของแต่ละ dimension -----
        # จะเห็นได้ว่า dimension ต่ำ → คลื่นถี่ (ความถี่สูง)
        #              dimension สูง → คลื่นห่าง (ความถี่ต่ำ)
        # นี่คือเหตุผลที่โมเดลแยกแยะตำแหน่งได้: แต่ละตำแหน่งมี combination ของ
        # คลื่นที่ความถี่ต่างกัน → ได้ "ลายเซ็น" เฉพาะตัว
        for dim in range(self.d_model):
            label = f"dim {dim} ({'sin' if dim % 2 == 0 else 'cos'})"
            axes[1].plot(range(seq_len), pe_2d[:, dim], marker="o", label=label)
        axes[1].set_xlabel("Position (ตำแหน่งคำ)")
        axes[1].set_ylabel("ค่า PE")
        axes[1].set_title("คลื่น sin/cos ในแต่ละ dimension\n(ความถี่ต่างกันทำให้แต่ละตำแหน่งไม่ซ้ำ)")
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # forward() = นำ position encoding ไปบวกกับ embedding ของคำ
    # นี่คือวิธี "ฉีด" ข้อมูลตำแหน่งเข้าไปใน input ก่อนส่งให้ Transformer
    # -------------------------------------------------------------------------
    def forward(self, x):
        # x มี shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        # บวกแบบ element-wise: embedding + position encoding
        # → ผลลัพธ์ยังคง shape เดิม แต่ตอนนี้แต่ละตำแหน่งมี "ตัวระบุ" ติดอยู่แล้ว
        return x + self.pe[:, :seq_len, :].to(x.device)
    
    # -------------------------------------------------------------------------
    # แสดง position encoding คู่กับ "คำจริง" ในประโยค
    # เพื่อให้เห็นภาพว่าแต่ละคำที่อยู่คนละตำแหน่ง จะได้ PE ต่างกันอย่างไร
    # (แสดงแค่ 4 dimension แรกเพื่อไม่ให้ตารางยาวเกินไป)
    # -------------------------------------------------------------------------
    def show_with_word(self, words):
        # หัวตาราง: ใช้ format spec แบบ '<N' (ชิดซ้าย กว้าง N ตัวอักษร)
        # หมายเหตุ: ห้ามมีช่องว่างระหว่าง '<' กับตัวเลข เช่น '< 40' จะ error
        print("-" * 70)
        print(f" {'POSITION ID ':<10} | {'word':<12} | {'position encoding (first 4 dims)':<40}")
        print("-" * 70)

        # self.pe มี shape (1, max_len, d_model)
        # → shape[1] คือจำนวนตำแหน่งสูงสุดที่เตรียมไว้ (max_len)
        max_positions = self.pe.shape[1]

        for i, word in enumerate(words):
            # ถ้ามีคำมากกว่า max_len ที่ pre-compute ไว้ ให้หยุด (ไม่งั้น index เกิน)
            if i >= max_positions:
                break

            # ดึง 4 dimension แรกของ position i
            # self.pe[0, i, :4] = batch 0, ตำแหน่งที่ i, dim 0..3
            vector = self.pe[0, i, :4]

            # แปลง tensor 4 ตัวเป็น string เช่น "+0.000 +1.000 +0.000 +1.000"
            # f"{v:+.3f}" = แสดงเครื่องหมาย +/- ตามด้วยทศนิยม 3 ตำแหน่ง
            vector_string = " ".join([f"{v:+.3f}" for v in vector])

            print(f" POSITION {i:<3} | {word:<12} | {vector_string:<40}")
        print("-" * 70)

# =============================================================================
# ทดลองใช้งาน
# =============================================================================
if __name__ == "__main__":
    # สร้าง position encoding ขนาด d_model=16, max_len=10
    # (จำลองว่ามีประโยคยาวสุด 10 คำ และแต่ละคำมี embedding 16 มิติ)
    positionEncoding = SinusoidalPositionEncoding(d_model=16, max_len=10)

    # 1) ดูแบบ tensor ดั้งเดิม
    positionEncoding.show_position_encoding()

    # 2) ดูแบบ DataFrame (อ่านง่ายกว่า เห็นแถว/คอลัมน์ชัดเจน)
    positionEncoding.show_as_dataframe()

    # 3) ดูแบบ heatmap + line plot (เห็น pattern คลื่นชัดเจนที่สุด)
    # positionEncoding.plot_heatmap()

    # 4) Wordlist
    word_list = ["I", "love", "NLP", "and", "Transformers", "are", "awesome", "!", "<PAD>", "<PAD>"]
    positionEncoding.show_with_word(word_list)


    # =========================================================================
    # 2) Scaled Dot-Product Attention + Padding Mask
    # -------------------------------------------------------------------------
    # หัวใจของ Transformer: ให้แต่ละคำ "มอง" คำอื่น ๆ ในประโยค แล้วดึงข้อมูล
    # ที่เกี่ยวข้องมาผสมเป็น representation ใหม่
    #
    # สูตร:  Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
    #
    #   - Q (Query) : "ฉันกำลังหาอะไร"
    #   - K (Key)   : "ฉันมีอะไรให้"
    #   - V (Value) : "ข้อมูลจริงที่จะส่งกลับ"
    #   - Q @ K^T   : วัดความเข้ากัน (similarity) ระหว่างทุกคู่ของคำ
    #   - / sqrt(d_k): scale ลดขนาด ป้องกัน softmax อิ่มตัว (saturate) เมื่อ d_k ใหญ่
    #   - softmax   : แปลงเป็น probability ที่รวมกันเท่ากับ 1 ในแต่ละแถว
    #   - @ V       : weighted sum ของ value vectors → ได้ context ใหม่
    # =========================================================================
    def scaled_dot_product_attention(query, key, value, mask=None):
        # query, key, value มี shape (batch_size, num_heads, seq_len, d_k)
        d_k = query.shape[-1]  # ขนาด vector ของแต่ละ head

        # ----- Step 1: คำนวณ attention score = Q @ K^T / sqrt(d_k) -----
        # key.swapaxes(-1, -2) สลับ 2 มิติสุดท้าย:
        #   (batch, heads, seq_len, d_k) → (batch, heads, d_k, seq_len)
        # หมายเหตุ: numpy ใช้ swapaxes/transpose() — ห้ามเขียนเป็น .transpose
        #          (ไม่มีวงเล็บ) เพราะนั่นคือ "อ้างถึงเมธอด" ไม่ได้เรียกใช้
        # ผลลัพธ์ shape: (batch, heads, seq_len, seq_len)
        # หมายความว่า score[b, h, i, j] = ความสนใจของคำที่ i ที่มีต่อคำที่ j
        score = (query @ key.swapaxes(-1, -2)) / np.sqrt(d_k)

        # ----- Step 2 (ถ้ามี mask): บล็อกตำแหน่งที่ไม่อยากให้สนใจ -----
        # mask = 0 → ตำแหน่งที่ไม่ควรมอง (เช่น <PAD>)
        # ใส่ค่าใกล้ -∞ เพื่อให้ softmax ให้น้ำหนัก ≈ 0
        if mask is not None:
            score = np.where(mask == 0, -1e9, score)

        # ----- Step 3: softmax ตามแกน -1 (ข้าม keys) -----
        # ลบ max ก่อน exp = เทคนิค numerical stability ป้องกัน overflow
        # ใช้ axis=-1 + keepdims=True เพื่อ normalize "ในแต่ละแถวของ query"
        # (คำหนึ่งคำให้ความสนใจกระจายไปยังคำอื่น ๆ รวมกันเท่ากับ 1)
        score_max = np.max(score, axis=-1, keepdims=True)
        weights = np.exp(score - score_max)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        # ----- Step 4: นำ attention weights ไปคูณกับ V -----
        # ผลลัพธ์ output shape: (batch, heads, seq_len, d_k)
        # = แต่ละคำได้ context ใหม่ที่เป็นค่าเฉลี่ยถ่วงน้ำหนักของ value ทุกคำ
        return weights @ value, weights
        
    # จำลองข้อมูล 1 ประโยค 3 tokens, vector size 4, 2 heads
    query = key = value = np.random.random((1, 2, 3, 4))  # shape (batch_size=1, num_heads=2, seq_len=3, d_k=4)
    output, weights = scaled_dot_product_attention(query, key, value)
    print("-" * 70)
    print(" Attention Weights (head 0) - shape 3 x 3 (seq_len x seq_len)")
    print(weights[0, 0].round(3))  # weights shape: (batch, heads, seq_len, seq_len) - head 0
    print("-" * 70)



# =============================================================================
# 3) Multi-Head Attention (การรวมหลายหัว attention เพื่อจับ pattern หลากหลาย)
# -----------------------------------------------------------------------------
# แทนที่จะคำนวณ attention เพียงรอบเดียว เราซอย vector ออกเป็นหลาย "หัว" (heads)
# แต่ละหัวเรียนรู้มองคำในมุมที่ต่างกัน เช่น:
#   - หัว 1: ความสัมพันธ์เชิงไวยากรณ์ (subject-verb)
#   - หัว 2: ความใกล้ทางความหมาย
#   - หัว 3: ตำแหน่งสัมพัทธ์ (ใกล้/ไกล)
#   - หัว 4: pattern ที่ยังไม่รู้ล่วงหน้า (โมเดลค้นเอง)
# แล้วเอาผลของทุกหัวมารวมกัน → ได้ representation ที่หลากหลายกว่า single-head
# =============================================================================

class MultiHeadAttention():
    """
    Multi-Head Attention (numpy version)
    ทำงาน 4 ขั้นตอน:
      1) project input → Q, K, V ด้วย weight matrices
      2) splitHeads: ซอย vector เป็นหลายหัว
      3) ทำ scaled_dot_product_attention ในทุกหัวพร้อมกัน (vectorized)
      4) combineHeads: รวมผลลัพธ์กลับ แล้วผ่าน W_o
    """
    def __init__(self, d_model=16, num_heads=4):
        # d_model    = ขนาด embedding ของแต่ละคำ (ต้องหาร num_heads ลงตัว)
        # num_heads  = จำนวนหัว attention
        # d_k        = ขนาดของแต่ละหัว = d_model / num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model ต้องหารด้วย num_heads ลงตัว"
        self.d_k = int(d_model // num_heads)  # การหารจำนวนเต็ม เศษตัดทิ้ง

        # weight matrices สำหรับ Q, K, V และ output projection
        # ใช้ numpy เพื่อให้สอดคล้องกับ scaled_dot_product_attention และ test data
        # คูณ 0.1 = scale ลงเพื่อให้ค่าเริ่มต้นไม่ใหญ่เกินไป (mimic Xavier init แบบง่าย)
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1  # Query projection
        self.W_k = np.random.randn(d_model, d_model) * 0.1  # Key projection
        self.W_v = np.random.randn(d_model, d_model) * 0.1  # Value projection
        self.W_o = np.random.randn(d_model, d_model) * 0.1  # Output projection (รวมหัว)

    def splitHeads(self, x):
        # ซอย d_model ออกเป็น num_heads หัว แต่ละหัวขนาด d_k
        # x shape:   (batch, seq_len, d_model)
        #   reshape: (batch, seq_len, num_heads, d_k)
        #   transpose(0,2,1,3) → (batch, num_heads, seq_len, d_k)
        # เหตุผลที่เอา num_heads ขึ้นมาก่อน seq_len:
        #   เพื่อให้ Q @ K^T ใน attention เกิดในแต่ละหัวอิสระจากกัน
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combineHeads(self, x):
        # ทำผกผันกับ splitHeads (รวมผลลัพธ์จากทุกหัวกลับเป็น d_model)
        # x shape:   (batch, num_heads, seq_len, d_k)
        #   transpose(0,2,1,3) → (batch, seq_len, num_heads, d_k)
        #   reshape: (batch, seq_len, num_heads * d_k) = (batch, seq_len, d_model)
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.d_k)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) — เป็น numpy array

        # ----- Step 1: สร้าง Q, K, V จาก input x -----
        # ใช้ matmul กับ weight matrices (ตามที่ครูสอน)
        # ผลลัพธ์ shape: (batch_size, seq_len, d_model)
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # ----- Step 2: แบ่ง Q, K, V ออกเป็นหลายหัว -----
        # ผลลัพธ์ shape: (batch_size, num_heads, seq_len, d_k)
        Q = self.splitHeads(Q)
        K = self.splitHeads(K)
        V = self.splitHeads(V)

        # ----- Step 3: คำนวณ attention พร้อมกันทุกหัว (vectorized) -----
        # scaled_dot_product_attention รองรับ batch dim หลายชั้นอยู่แล้ว
        # → ไม่ต้อง for loop ทีละหัว (เร็วกว่ามาก)
        # attn_output shape:  (batch_size, num_heads, seq_len, d_k)
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)

        # ----- Step 4: รวมผลลัพธ์จากทุกหัว แล้วผ่าน W_o -----
        # combined shape: (batch_size, seq_len, d_model)
        combined = self.combineHeads(attn_output)
        output = np.matmul(combined, self.W_o)
        return output, attn_weights

# ทดสอบรัน จำลอง Input Size 16D แบ่งเป็น 4 Head, Head ละ 4 Dimension
InputData = np.random.randn(1, 5, 16)
mha = MultiHeadAttention()
heads = mha.splitHeads(InputData)
print("-" * 70)
print(f"- Origin Shape: {InputData.shape} \n- Heads Shape : {heads.shape} (Batch, Head, Seq_len, Depth)")
print("-" * 70)

# =============================================================================
# อธิบายแบบเข้าใจง่าย: ทำไม Shape ถึงเป็นแบบนี้
# =============================================================================
#
# 🔹 Origin Shape: (1, 5, 16)  ← มาจาก np.random.randn(1, 5, 16)
#   - 1  = batch_size  → มี 1 ประโยค (1 ก้อนข้อมูล)
#   - 5  = seq_len     → ประโยคนี้มี 5 คำ (5 tokens)
#   - 16 = d_model     → แต่ละคำถูกแปลงเป็น vector ขนาด 16 มิติ
#   เปรียบเทียบ: เหมือนตาราง 5 แถว (คำ) × 16 คอลัมน์ (ความหมายของคำ)
#
# 🔹 Heads Shape: (1, 4, 5, 4)  ← หลังผ่าน splitHeads()
#   - 1 = batch_size   → ยังเป็น 1 ประโยคเหมือนเดิม
#   - 4 = num_heads    → ซอย vector 16 มิติ ออกเป็น 4 หัว (heads)
#   - 5 = seq_len      → ยังคง 5 คำเหมือนเดิม
#   - 4 = d_k          → แต่ละหัวถือ vector ขนาด 4 มิติ (16 ÷ 4 = 4)
#
# 🔹 ขั้นตอนการแปลงร่าง (reshape + transpose):
#   (1, 5, 16)
#       │  reshape เป็น (batch, seq_len, num_heads, d_k)
#       ▼
#   (1, 5, 4, 4)         ← แบ่ง 16 มิติ → 4 หัว × 4 มิติ
#       │  transpose(0, 2, 1, 3)  สลับแกน seq_len ↔ num_heads
#       ▼
#   (1, 4, 5, 4)         ← (Batch, Head, Seq_len, Depth)
#
# 🔹 ทำไมต้องสลับให้ Head มาก่อน Seq_len?
#   เพราะตอนคำนวณ attention เราต้องการให้ "แต่ละหัว" ทำงาน
#   อิสระกับคำทั้ง 5 คำ → เลยจัดให้ head เป็นแกนนอกสุด
#   เพื่อให้ Q @ K^T เกิดขึ้นภายในหัวเดียวกัน ไม่ปนข้ามหัว
#
# 🔹 อุปมา: เหมือนแบ่งทีมนักสืบ 4 คน ดูประโยคเดียวกัน
#   แต่ละคนสนใจมุมที่ต่างกัน (ไวยากรณ์ / อารมณ์ / ความสัมพันธ์ ฯลฯ)
#   แล้วค่อยเอาผลลัพธ์มารวมกันตอนท้าย

# =============================================================================
# รวม Positional Encoding + Multi-Head Attention เข้าด้วยกัน (mini pipeline)
# =============================================================================
d_model = 16
num_heads = 4
seq_len = 5
batch = 1

# Step 1: สุ่ม token embeddings (จำลอง embedding ของ 5 คำ)
np.random.seed(0)
token_embeddings = np.random.randn(batch, seq_len, d_model)

# Step 2: บวกกับ Positional Encoding เพื่อ "ฉีด" ข้อมูลตำแหน่งเข้าไป
position_Encoding = SinusoidalPositionEncoding(d_model=d_model, max_len=10)
position_Encoding.show_position_encoding(seq_len=seq_len)
# pe เป็น torch tensor shape (1, max_len, d_model) → แปลงเป็น numpy + slice ตามจำนวนคำจริง
pe_np = position_Encoding.pe.detach().numpy()[:, :seq_len, :]  # shape (1, seq_len, d_model)
Input = token_embeddings + pe_np

# Step 3: ส่งเข้า Multi-Head Attention
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
output, attn_weights = mha.forward(Input)

print("-" * 70)
print(f"Input shape         : {Input.shape}          (batch, seq_len, d_model)")
print(f"Output shape        : {output.shape}          (batch, seq_len, d_model)")
print(f"Attn weights shape  : {attn_weights.shape}    (batch, num_heads, seq_len Q , seq_len k)")
print("-" * 70)

# -----------------------------------------------------------------------------
# แสดง attention weight ของแต่ละหัวแยกกัน
# -----------------------------------------------------------------------------
# attn_weights shape: (batch, num_heads, seq_len, seq_len)
# - แถว i = "คำที่ i" กำลังให้ความสนใจ (query)
# - คอลัมน์ j = "คำที่ j" ที่ถูกมอง (key)
# - ค่า [i, j] = ความสนใจของคำ i ที่มีต่อคำ j (รวมแต่ละแถว = 1)
#
# การแสดงทีละหัวช่วยให้เห็นว่า "แต่ละหัวสนใจคนละ pattern"
# (ในตัวอย่างนี้ weight ยังกระจายค่อนข้างเท่ากัน เพราะ W_q/W_k/W_v เป็นค่าสุ่ม
#  ยังไม่ได้ train — ของจริงแต่ละหัวจะมี pattern แตกต่างกันชัดเจน)
# -----------------------------------------------------------------------------
words = ["I", "love", "NLP", "and", "!"]  # ตัวอย่างคำสำหรับ 5 ตำแหน่ง

print("Attention Weights per Head  (each row sums to 1)")
print("Row = Query (the word looking) | Col = Key (the word being looked at)")
print("=" * 70)

for h in range(num_heads):
    head_weight = attn_weights[0, h].round(3)  # shape (seq_len, seq_len)

    print(f"\n[ Head {h} ]")
    # หัวตาราง: ชื่อคำเป็น Key (คอลัมน์)
    header = "        " + "".join([f"{w:>8}" for w in words])
    print(header)
    print("        " + "-" * (8 * len(words)))

    # แต่ละแถว: ชื่อคำเป็น Query
    for i, w in enumerate(words):
        row_str = f"{w:>6} |" + "".join([f"{v:>8.3f}" for v in head_weight[i]])
        print(row_str)

print("=" * 70)

# Tranformer Endcoder (BERT) and Decode (GPT)
class FeedForWard:
    def __init__(self, d_model, dimention_feedforward=None, seed=42):
        rng = np.random.RandomState(seed)
        dimention_feedforward = dimention_feedforward or d_model * 4
        s = np.sqrt(2.0 / d_model)
        s = np.sqrt(2.0 / d_model)
        self.W1 = rng.randn(dimention_feedforward, d_model) * s       # (dimention_feedforward, d_model)
        self.b1 = np.zeros(dimention_feedforward)
        self.W2 = rng.randn(d_model, dimention_feedforward) * s       # (d_model, dimention_feedforward)
        self.b2 = np.zeros(d_model)