# One sequence Multi-Head Attention
import math
import random

# ---------- 基本運算 ----------
def matmul(A, B):                      # (n,k) @ (k,m) -> (n,m)
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]

def transpose(A):
    return [list(r) for r in zip(*A)]

def softmax(row):
    m = max(row)                       # 減最大值,避免 exp 溢位
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]

# ---------- Scaled Dot-Product Attention ----------
def attention(Q, K, V, mask=None):
    d_k = len(Q[0])
    scores = matmul(Q, transpose(K))                       # (seq, seq)
    scores = [[s / math.sqrt(d_k) for s in row] for row in scores]
    if mask is not None:                                   # mask[i][j]=False 的位置不能看
        scores = [[s if mask[i][j] else -1e9 for j, s in enumerate(row)]
                  for i, row in enumerate(scores)]
    weights = [softmax(row) for row in scores]
    return matmul(weights, V)                              # (seq, d_k)

# ---------- 切 head / 合併 head ----------
def split_heads(X, h):                 # (seq, d_model) -> h 個 (seq, d_k)
    d_k = len(X[0]) // h
    return [[row[i * d_k:(i + 1) * d_k] for row in X] for i in range(h)]

def concat_heads(heads):               # h 個 (seq, d_k) -> (seq, d_model)
    return [sum((head[t] for head in heads), []) for t in range(len(heads[0]))]

# ---------- Multi-Head Attention ----------
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.h = num_heads
        s = 1 / math.sqrt(d_model)
        rand = lambda: [[random.uniform(-s, s) for _ in range(d_model)]
                        for _ in range(d_model)]
        self.W_q, self.W_k, self.W_v, self.W_o = rand(), rand(), rand(), rand()

    def __call__(self, x, mask=None):
        Q, K, V = matmul(x, self.W_q), matmul(x, self.W_k), matmul(x, self.W_v)
        heads = [attention(q, k, v, mask)
                 for q, k, v in zip(split_heads(Q, self.h),
                                    split_heads(K, self.h),
                                    split_heads(V, self.h))]
        return matmul(concat_heads(heads), self.W_o)

# ---------- 測試 ----------
if __name__ == "__main__":
    seq_len, d_model, num_heads = 4, 8, 2
    x = [[random.gauss(0, 1) for _ in range(d_model)] for _ in range(seq_len)]
    causal = [[j <= i for j in range(seq_len)] for i in range(seq_len)]  # decoder 用

    out = MultiHeadAttention(d_model, num_heads)(x, mask=causal)
    print(len(out), len(out[0]))       # 4 8,shape 跟輸入相同
