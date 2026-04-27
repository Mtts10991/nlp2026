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
        print("" *70)
        print(f"Position Encoding for sequence length {seq_len} tokens (shape: {self.pe.shape}):")
        print("" *70)
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
    query = key = value = np.random.random((1, 2, 3, 4))  # shape (batch_size=1, num_heads=2, seq_len=3, d_k=4
    output, weights = scaled_dot_product_attention(query, key, value)
    print("" * 70)
    print(f" Attention Weights 3 X 3 Matrixs")
    print(weights[0].round(3))  # แสดง weight matrix ของ head 0 (shape 3x3)
    print("" * 70)
