import math


# ---------- 原本的 RMSNorm ----------
def rms_norm(x, gamma, eps=1e-6):   # x: 單一 token 的向量
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [g * v / rms for g, v in zip(gamma, x)]


# ---------- Gemma 4 風格 RMSNorm (Gemma4RMSNorm) ----------
# weight 初始為 1,輸出 = x / rms * weight
# weight=None 對應 with_scale=False (Gemma 4 的 V-Norm: 只 normalize,沒有可學習 scale)
def gemma_rms_norm(x, weight=None, eps=1e-6):
    inv = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    if weight is None:
        return [v * inv for v in x]
    return [v * inv * w for v, w in zip(x, weight)]


# ---------- Qwen3.5 風格 RMSNorm (Qwen3_5RMSNorm) ----------
# weight 初始為 0,輸出 = x / rms * (1 + weight)
# 初始行為與 gemma_rms_norm(weight=全 1) 相同,只是參數化不同
def qwen_rms_norm(x, weight, eps=1e-6):
    inv = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    return [v * inv * (1.0 + w) for v, w in zip(x, weight)]


# ---------- 對整個 (seq, dim) 矩陣逐 token 套用 norm ----------
def apply_norm(X, norm_fn):
    return [norm_fn(row) for row in X]


def add(A, B):                          # 殘差相加
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


# ---------- QK-Norm ----------
# Q, K: (seq, d_model) 先切 head 後,每個 head 的 d_k 維向量各自 normalize
# weight 長度是 d_k (不是 d_model)
def qk_norm(X, num_heads, norm_fn):
    d_k = len(X[0]) // num_heads
    out = []
    for row in X:
        new_row = []
        for h in range(num_heads):
            new_row += norm_fn(row[h * d_k:(h + 1) * d_k])
        out.append(new_row)
    return out


# ---------- Decoder layer 的 norm 位置 ----------
# attn / mlp: 輸入輸出皆為 (seq, d_model) 的函式
# norms: dict,每個值是「單一 token 向量 -> 向量」的 norm 函式

def pre_norm_layer(X, attn, mlp, norms):
    """Qwen3.5 / LLaMA: 每層 2 個 norm,都在子層輸入端。
    x = x + Attn(Norm(x))
    x = x + MLP(Norm(x))
    """
    X = add(X, attn(apply_norm(X, norms["input_layernorm"])))
    X = add(X, mlp(apply_norm(X, norms["post_attention_layernorm"])))
    return X


def sandwich_layer(X, attn, mlp, norms):
    """Gemma 4: 每層 4 個 norm,子層前後各一個,post norm 在殘差相加之前。
    x = x + PostNorm(Attn(PreNorm(x)))
    x = x + PostNorm(MLP(PreNorm(x)))
    """
    h = attn(apply_norm(X, norms["input_layernorm"]))
    X = add(X, apply_norm(h, norms["post_attention_layernorm"]))
    h = mlp(apply_norm(X, norms["pre_feedforward_layernorm"]))
    X = add(X, apply_norm(h, norms["post_feedforward_layernorm"]))
    return X


# ---------- 測試 ----------
if __name__ == "__main__":
    import random

    d = 6
    x = [random.gauss(0, 1) for _ in range(d)]

    # 1. 初始值下,Qwen (weight=0, 乘 1+w) 與 Gemma (weight=1) 結果相同
    q = qwen_rms_norm(x, [0.0] * d)
    g = gemma_rms_norm(x, [1.0] * d)
    assert all(abs(a - b) < 1e-9 for a, b in zip(q, g))

    # 2. 與原本的 rms_norm 一致
    r = rms_norm(x, [1.0] * d)
    assert all(abs(a - b) < 1e-9 for a, b in zip(r, g))

    # 3. 沒有 scale 的版本 (V-Norm): normalize 後 RMS ≈ 1
    v = gemma_rms_norm(x)
    print("V-Norm 後的 RMS:", math.sqrt(sum(t * t for t in v) / d))

    # 4. 兩種 layer 排列,shape 保持 (seq, d_model)
    seq = 3
    X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq)]
    identity = lambda M: M                      # 用恆等函式代替 attn / mlp,只看 norm 位置

    qwen_norms = {
        "input_layernorm": lambda t: qwen_rms_norm(t, [0.0] * d),
        "post_attention_layernorm": lambda t: qwen_rms_norm(t, [0.0] * d),
    }
    gemma_norms = {
        k: (lambda t: gemma_rms_norm(t, [1.0] * d))
        for k in ("input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm")
    }
    print(len(pre_norm_layer(X, identity, identity, qwen_norms)[0]))     # 6
    print(len(sandwich_layer(X, identity, identity, gemma_norms)[0]))    # 6

    # 5. QK-Norm: 2 個 head,每個 head 的 3 維向量各自 normalize
    Q = qk_norm(X, 2, lambda t: gemma_rms_norm(t, [1.0] * 3))
    print(len(Q), len(Q[0]))                                              # 3 6
    print("OK")
