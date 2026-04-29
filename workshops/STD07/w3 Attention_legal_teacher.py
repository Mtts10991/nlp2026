import numpy as np


# 1. Sinusoidal Position Encoding การระบุข้อมูลตำแหน่ง เพื่อให้โมเดลเข้าใจลำดับคำ
class SinusoidalPositionEncoding: 
    def __init__(self, max_seq_len = 10, d_model = 16):
        pe = np.zeros((max_seq_len,d_model))
        pos = np.arange(max_seq_len).reshape(-1,1)
        div =np.power(10000.0 , np.arange(0,d_model,2)/d_model)
        pe[:,0::2] = np.sin(pos/div)
        pe[:, 1::2] = np.cos(pos/div)
        self.pe = pe

    def encode(self, X):
        # บวก position encoding เข้ากับ input X
        # รองรับทั้ง 2D (seq, d_model) และ 3D (batch, seq, d_model)
        seq_len = X.shape[-2]
        return X + self.pe[:seq_len]

    def show(self,seq_len=5):
        print("" * 70)
        print(f"Position Encoding (First {seq_len} tokens)")
        for p in range(seq_len):
            print(f"Pos : {p}"+" ".join(f"{v:.2f}" for v in self.pe[p, :10])+"...")

    def show_with_words(self,words):
        print("" * 70)
        print(f"{"index" : <7} | {'word' : <10} | {'Positional Encoding (First 4 dims)' :<40}")     
        for i, word in enumerate(words):
            if i >= len(self.pe):
                break
            vec = self.pe[i,:10]
            vec_str = " ".join(f"{v:+.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word:<10}| [{vec_str}...]")
#การใช้งาน
words_list = ["I", "love","learning", "AI", "Technology"]
pe = SinusoidalPositionEncoding(max_seq_len=10, d_model=16)
pe.show_with_words(words_list)
pe = SinusoidalPositionEncoding()
pe.show()

# 2. Scale Dot-Product and Padding Mask ให้ความยาวของเอกสารเท่ากัน
def scaled_dot_product_attention(Q,K,V, mask=None):
    d_k =Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0,2,1))/ np.sqrt(d_k)
    if mask is not None:
        # กำหนดให้ตำแหน่งที่เป็น 0 ใน mask มีค่า score เป็น -inf
        scores = np.where(mask==0, -1e9, scores)
    weights = np.exp(scores-np.max(scores))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights
# จำลองข้อมูล 1 ประโยค 3 Tokens , vector 4 มิติ
q = k = v = np.random.randn(1,3,4)
output , weights = scaled_dot_product_attention(q,k,v)
print("=" * 70)
print(f"---Attention Weights 3x3 Matrices ----")
print(weights[0].round(2))
print("=" * 70)

# 3.1 Multi-Head Attention (การมองหลายมุมมอง) ช่วยให้โมเดลเข้าใจความสัมพันธ์หลายรูปแบบพร้อมกัน
class MultiHeadAttentionSimple:
    def __init__(self, d_model=16 ,n_heads = 4, seed=42):
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

        self.n_heads = n_heads
        self.d_k = int(d_model // n_heads)
    def split_heads(self, x):
        batch , seq_len, d_model = x.shape
        x = x.reshape(batch,seq_len,self.n_heads,self.d_k)
        return x.transpose(0,2,1,3)

    def combine_heads(self, x): # การรวม Head กลับ (Batch, heads, seq, d_k) -> (batch , seq ,d_model)
        batch, heads, seq_len, d_k = x.shape
        x = x.transpose(0,2,1,3)
        return x.reshape(batch, seq_len ,heads*d_k)
    
    def forward(self, x, padding_mask=None, causal_mask=None, return_weights=False):
        # รองรับ input 2D (seq, d_model) — เพิ่ม batch dim ชั่วคราว
        squeeze_batch = (x.ndim == 2)
        if squeeze_batch:
            x = x[np.newaxis, :, :]

        # 1. Linear projection ก่อนแยก Heads (1,5,16)
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        # 2 แยกออกเป็น 4 หัว (1,4,5,4)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # 3 Attention แต่ละ Head-reshape เพื่อให้ batch Head รวมกัน
        batch = x.shape[0]
        seq = x.shape[1]
        Q_r = Q.reshape(batch * self.n_heads, seq, self.d_k) # batch 1 -> (4,5,4)
        K_r = K.reshape(batch * self.n_heads, seq, self.d_k)
        V_r = V.reshape(batch * self.n_heads, seq, self.d_k)

        # 3.5 รวม causal_mask + padding_mask ให้เป็น mask เดียว shape (batch*heads, seq, seq)
        # convention: 1 = keep, 0 = block (ตามที่ scaled_dot_product_attention ใช้)
        mask = None
        if causal_mask is not None or padding_mask is not None:
            mask = np.ones((batch * self.n_heads, seq, seq), dtype=np.float32)
            if causal_mask is not None:
                # causal_mask: (seq, seq) bool, True = blocked
                cm = (~causal_mask).astype(np.float32)
                mask = mask * cm
            if padding_mask is not None:
                pm = np.asarray(padding_mask, dtype=np.float32)
                if pm.ndim == 1:
                    pm = pm[np.newaxis, :]
                pm_b = np.broadcast_to(pm[:, np.newaxis, np.newaxis, :],
                                       (batch, self.n_heads, seq, seq))
                mask = mask * pm_b.reshape(batch * self.n_heads, seq, seq)

        attn_out,attn_weights = scaled_dot_product_attention(Q_r,K_r,V_r, mask=mask)
        # Reshape Attention weight -> (batch , heads , seq , seq)
        attn_weights = attn_weights.reshape(batch,self.n_heads, seq, seq)
        #Step 4: รวม Heads กลับ
        attn_out = attn_out.reshape(batch,self.n_heads, seq, self.d_k)
        concat = self.combine_heads(attn_out)
        # Step 5 :Output Projection
        output = np.matmul(concat,self.W_o)

        # ถ้า input เดิมเป็น 2D ให้ตัด batch dim ออกก่อน return
        if squeeze_batch:
            output = output[0]                  # (seq, d_model)
            attn_weights = attn_weights[0]      # (heads, seq, seq)

        return output , attn_weights
    
# จำลอง Input ขนาด 16 - มิติ  (d) แบ่งเป็น 4 Head (Head ละ 4  dim)    
input_data = np.random.randn(1,5,16)
mha = MultiHeadAttentionSimple()
heads = mha.split_heads(input_data)
print(f"Origin Shape :{input_data.shape}")
print(f"Heads shape :{heads.shape} (Batch , Heads , Seq_len , Depth)")

# 3.2  Positional Encoding Multi-Head Attention
d_model = 16
n_heads = 4
seq_len = 5
batch = 1
# Step1 : Random Input Embedding
np.random.seed(0)
token_embeddings = np.random.randn(batch,seq_len,d_model)
#Step2 : บวก Positional Encoding
pe_encoder = SinusoidalPositionEncoding(max_seq_len=10, d_model=d_model)
pe_encoder.show(seq_len)
x = token_embeddings + pe_encoder.pe[:seq_len]
#Step3 : Multi Head Attention
mha = MultiHeadAttentionSimple(d_model=d_model, n_heads=n_heads)
output, attn_weights = mha.forward(x)

#แสดงผล
print(f"--MultiHead Attention (4 Heads)--")
print(f"Input Shape : {x.shape}")
print(f"Output Shape : {output.shape}")
print(f"Weight Shape : {attn_weights.shape} -> (batch, heads, seq_q, seq_k)")
for h in range(n_heads):
    print(f"\nHead {h+1} attention weight:")
    for row in attn_weights[0,h]:
        print(" " , " ".join(f"{v: .3f}" for v in row))


# ============================================================
# PART 1: Feed-Forward Network (FFN) — ส่วนที่ขาดหายใน w3
# ============================================================

class FeedForward:
    """
    Position-wise FFN: FFN(x) = ReLU(xW1 + b1)W2 + b2
    d_model → d_ff (ปกติ 4×d_model) → d_model
    """
    def __init__(self, d_model, d_ff=None, seed=42):
        rng = np.random.RandomState(seed)
        d_ff = d_ff or d_model * 4
        s = np.sqrt(2.0 / d_model)
        self.W1 = rng.randn(d_ff, d_model) * s       # (d_ff, d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_model, d_ff) * s       # (d_model, d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        h = np.maximum(0, X @ self.W1.T + self.b1)   # ReLU
        return h @ self.W2.T + self.b2


def layer_norm(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)


# ============================================================
# PART 2: Transformer Encoder Block (BERT-style)
# ============================================================

class TransformerEncoderBlock:
    """
    BERT-style Encoder Block — Pre-Norm Architecture (v3-1)

    Forward Pass:
        x1 = x + MHA( LayerNorm(x) )          ← Bidirectional (ไม่มี causal mask)
        x2 = x1 + FFN( LayerNorm(x1) )
        return x2

    คุณสมบัติ Encoder:
        - มองเห็นทุก token ทั้งซ้ายและขวา (Full Attention)
        - เหมาะงาน: Classification, NER, Q&A (เหมือน BERT)
    """

    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.mha = MultiHeadAttentionSimple(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed+1)

    def forward(self, X, padding_mask=None, return_weights=False):
        # Sub-layer 1: Multi-Head Attention (ไม่มี causal mask)
        attn_out, hw = self.mha.forward(
            X, padding_mask=padding_mask,
            causal_mask=None,              # ← BERT: มองเห็นทุกทิศ
            return_weights=return_weights
        )
        x1 = X + attn_out                 # Residual connection

        # Sub-layer 2: FFN
        x2 = x1 + self.ffn.forward(layer_norm(x1))

        return x2, hw


# ============================================================
# PART 3: Transformer Decoder Block (GPT-style)
# ============================================================

class TransformerDecoderBlock:
    """
    GPT-style Decoder Block — Pre-Norm + Causal Mask (v3-1, v3-2)

    Forward Pass:
        x1 = x + MHA( LayerNorm(x), causal_mask=True )  ← มองได้แค่ซ้าย
        x2 = x1 + FFN( LayerNorm(x1) )
        return x2

    คุณสมบัติ Decoder:
        - token ที่ตำแหน่ง t มองได้เฉพาะ t0..t (ไม่มองอนาคต)
        - เหมาะงาน: Text Generation (เหมือน GPT)
    """

    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.mha = MultiHeadAttentionSimple(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed+1)

    @staticmethod
    def _causal_mask(T):
        """Upper-triangular True = blocked (ดูสูตรใน TransformerEncoderVsDecoder.causal_mask)"""
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    def forward(self, X, padding_mask=None, return_weights=False):
        T = X.shape[0]
        mask = self._causal_mask(T)        # (T, T) — เหมือนที่ w3 ใช้

        # Sub-layer 1: Masked Multi-Head Attention (GPT: มองแค่ซ้าย)
        attn_out, hw = self.mha.forward(
            X, padding_mask=padding_mask,
            causal_mask=mask,              # ← GPT: บัง token อนาคต
            return_weights=return_weights
        )
        x1 = X + attn_out

        # Sub-layer 2: FFN
        x2 = x1 + self.ffn.forward(layer_norm(x1))

        return x2, hw


# ============================================================
# PART 3.5: Encoder vs Decoder — เปรียบเทียบ Attention Mask
# ============================================================

class TransformerEncoderVsDecoder:
    """
    Utility class สำหรับเปรียบเทียบ attention mask ระหว่าง:
      - Encoder (BERT) → Full attention (ทุก token เห็นกันได้หมด)
      - Decoder (GPT)  → Causal mask (มองได้แค่ token ก่อนหน้า)
    """
    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.seed = seed

    @staticmethod
    def causal_mask(T):
        # Upper-triangular True = blocked (เหมือนของ Decoder block)
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    def print_comparison(self, T=4):
        # Encoder: ไม่มี mask — เห็นกันหมด
        print(f"\n  Encoder (BERT) — Full Attention (no mask):")
        print(f"  pos  " + "".join(f"  t{j}" for j in range(T)))
        for i in range(T):
            print(f"   t{i}  " + "  v" * T)

        # Decoder: causal mask
        print(f"\n  Decoder (GPT) — Causal Mask:")
        print(f"  pos  " + "".join(f"  t{j}" for j in range(T)))
        cm = self.causal_mask(T)
        for i in range(T):
            row = "".join("  v" if not cm[i, j] else "  x" for j in range(T))
            print(f"   t{i}  {row}")


# ============================================================
# PART 4: Mini BERT (Encoder-only, 2 layers)
# ============================================================

class MiniBERT:
    """
    BERT-style Encoder: Embed + PE → [EncoderBlock × n_layers] → mean pool → classify
    ใช้ Full Attention ทุก layer
    """

    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, n_classes=3, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / input_size)
        self.W_proj = rng.randn(d_model, input_size) * s      # Embedding projection
        self.pe = SinusoidalPositionEncoding(512, d_model)
        self.layers = [
            TransformerEncoderBlock(d_model, n_heads, seed=seed + i)
            for i in range(n_layers)
        ]
        self.W_out = rng.randn(n_classes, d_model) * s
        self.b_out = np.zeros((n_classes, 1))

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x_seq, padding_mask=None, return_weights=False):
        X = x_seq @ self.W_proj.T                    # Project input
        X = self.pe.encode(X)                        # บวก PE (เหมือน w3)

        all_weights = []
        for layer in self.layers:
            X, hw = layer.forward(X, padding_mask=padding_mask,
                                  return_weights=return_weights)
            if return_weights:
                all_weights.append(hw)

        ctx = X.mean(axis=0)                         # Mean pooling
        probs = self._softmax(
            (self.W_out @ ctx.reshape(-1, 1) + self.b_out).flatten()
        )
        return probs, all_weights

    def predict_batch(self, X, sl):
        return np.array([np.argmax(self.forward(X[i].reshape(sl, -1))[0]) for i in range(len(X))])

    def predict_proba(self, X, sl):
        return np.array([self.forward(X[i].reshape(sl, -1))[0] for i in range(len(X))])


# ============================================================
# PART 5: Mini GPT (Decoder-only, 2 layers)
# ============================================================

class MiniGPT:
    """
    GPT-style Decoder: Embed + PE → [DecoderBlock × n_layers] → last token → next token pred
    ใช้ Causal Mask ทุก layer (มองแค่ token ก่อนหน้า)
    """

    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / input_size)
        self.W_proj = rng.randn(d_model, input_size) * s
        self.pe = SinusoidalPositionEncoding(512, d_model)
        self.layers = [
            TransformerDecoderBlock(d_model, n_heads, seed=seed + i)
            for i in range(n_layers)
        ]
        # Language model head: predict next token (project back to input_size)
        self.W_lm = rng.randn(input_size, d_model) * s

    def forward(self, x_seq, return_weights=False):
        X = x_seq @ self.W_proj.T
        X = self.pe.encode(X)

        all_weights = []
        for layer in self.layers:
            X, hw = layer.forward(X, return_weights=return_weights)
            if return_weights:
                all_weights.append(hw)

        # GPT: ใช้เฉพาะ hidden state ของ token สุดท้ายในการทำนาย
        last_hidden = X[-1]                          # (d_model,)
        next_token_logits = self.W_lm @ last_hidden  # (input_size,)

        return next_token_logits, all_weights


# ============================================================
# MAIN DEMO
# ============================================================

def print_section(title, char="="):
    print(f"\n{char*58}\n  {title}\n{char*58}")


def _print_heatmap(W, seq_len):
    # พิมพ์ attention matrix (seq, seq) แบบอ่านง่าย
    for i in range(seq_len):
        print("  " + " ".join(f"{v: .3f}" for v in W[i]))


def run_demo():
    d_model   = 32
    n_heads   = 4
    seq_len   = 4
    input_dim = 8      # ขนาด feature ต่อ token

    rng = np.random.RandomState(0)
    X_sample = rng.randn(seq_len, input_dim) * 0.1   # (4, 8) ทดสอบ

    # ─── STEP 1: w3 TransformerEncoderVsDecoder (ของเดิม) ───────────
    print_section("STEP 1: TransformerEncoderVsDecoder (จาก w3)")
    ev = TransformerEncoderVsDecoder(d_model=d_model, n_heads=n_heads, seed=42)
    ev.print_comparison(T=seq_len)

    # ─── STEP 2: Encoder Block ───────────────────────────────────────
    print_section("STEP 2: Encoder Block (BERT-style) — Full Attention")
    enc_block = TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, seed=42)

    # Project input ก่อน (เหมือน MiniBERT)
    W_proj = rng.randn(d_model, input_dim) * 0.1
    pe = SinusoidalPositionEncoding(512, d_model)
    X_emb = pe.encode(X_sample @ W_proj.T)

    enc_out, enc_ws = enc_block.forward(X_emb, return_weights=True)
    print(f"\n  Input  shape : {X_emb.shape}")
    print(f"  Output shape : {enc_out.shape}  (shape เท่ากับ input — residual)")
    print(f"\n  Encoder Head 1 — Full Attention (ทุก token มองเห็นกัน):")
    _print_heatmap(enc_ws[0], seq_len)

    # ─── STEP 3: Decoder Block ───────────────────────────────────────
    print_section("STEP 3: Decoder Block (GPT-style) — Causal Mask")
    dec_block = TransformerDecoderBlock(d_model=d_model, n_heads=n_heads, seed=42)
    dec_out, dec_ws = dec_block.forward(X_emb, return_weights=True)

    print(f"\n  Decoder Head 1 — Causal Masked (บัง token อนาคต):")
    _print_heatmap(dec_ws[0], seq_len)

    print(f"\n  Causal Mask pattern:")
    mask = TransformerDecoderBlock._causal_mask(seq_len)
    print("  pos  " + "".join(f"  t{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = "".join("  ✓" if not mask[i, j] else "  ✗" for j in range(seq_len))
        print(f"   t{i}  {row}")

    # ─── STEP 4: MiniBERT (2 layers) ─────────────────────────────────
    print_section("STEP 4: MiniBERT — 2-Layer Encoder")
    bert = MiniBERT(input_size=input_dim, d_model=d_model,
                    n_heads=n_heads, n_layers=2, n_classes=3, seed=42)
    probs, bert_ws = bert.forward(X_sample, return_weights=True)
    print(f"\n  Class probabilities: {probs.round(4)}")
    print(f"  Predicted class    : {np.argmax(probs)}")
    print(f"\n  Layer 1 — Head 1 attention:")
    _print_heatmap(bert_ws[0][0], seq_len)
    print(f"\n  Layer 2 — Head 1 attention:")
    _print_heatmap(bert_ws[1][0], seq_len)

    # ─── STEP 5: MiniGPT (2 layers) ──────────────────────────────────
    print_section("STEP 5: MiniGPT — 2-Layer Decoder")
    gpt = MiniGPT(input_size=input_dim, d_model=d_model,
                  n_heads=n_heads, n_layers=2, seed=42)
    logits, gpt_ws = gpt.forward(X_sample, return_weights=True)
    print(f"\n  Next-token logits (8 dims): {logits.round(4)}")
    print(f"\n  Layer 1 — Head 1 attention (Causal):")
    _print_heatmap(gpt_ws[0][0], seq_len)
    print(f"\n  Layer 2 — Head 1 attention (Causal):")
    _print_heatmap(gpt_ws[1][0], seq_len)

    # ─── STEP 6: สรุปความต่าง ─────────────────────────────────────────
    print_section("STEP 6: Encoder vs Decoder — สรุป")
    print("""
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │                     │  Encoder (BERT)       │  Decoder (GPT)       │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ Attention Mask      │ Full (ไม่มี mask)       │ Causal (บังอนาคต)      │
  │ token t มองเห็น      │ ทุก token              │ t0 .. t เท่านั้น         │
  │ Output ใช้ทำอะไร      │ Mean pool → classify  │ Last token → predict │
  │ งานหลัก              │ Classification, NER    │ Text generation      │
  │ ตัวอย่าง Model        │ BERT, RoBERTa           │ GPT-2, GPT-4         │
  │ Pre-Norm (v3-1)     │ ✅                      │ ✅                   │
  │ Padding Mask (v3-2) │ ✅                      │ —                    │
  └─────────────────────┴──────────────────────────┴──────────────────────┘
    """)
run_demo()
