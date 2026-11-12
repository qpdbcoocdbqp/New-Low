# Mixture-of-Experts (MoE) layer:取代 Transformer block 裡的 FFN(attention 不動)
# 流程:router 打分 -> 每個 token 選 top-k 個 expert -> expert 輸出按權重加總
#
# 各 LLM 的差異只在三件事:
#   1. router 怎麼打分 / 選 / 算權重   2. shared expert 怎麼接   3. expert 內的激活函數
# 下面每個版本都標了來源:
#   [src] = 對照 transformers 5.17.0 的 modeling_*.py 確認過
#   [?]   = 沒有原始碼可對照,是推測或來自網頁資料
import math
import random

# ---------- 基本運算 ----------
def matmul(A, B):                      # (n,k) @ (k,m) -> (n,m)
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]

def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def silu(x):
    return x * sigmoid(x)

def gelu_tanh(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def sqrtsoftplus(x):                   # DeepSeek V4 的 scoring func:sqrt(softplus(x))
    return math.sqrt(math.log1p(math.exp(x)) if x < 30 else x)

def rms_norm(x, gamma=None, eps=1e-6):  # 單一 token 向量;gamma=None 表示不帶 scale
    r = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / r * (gamma[i] if gamma else 1.0) for i, v in enumerate(x)]

def rand_matrix(n, m):
    s = 1 / math.sqrt(n)
    return [[random.uniform(-s, s) for _ in range(m)] for _ in range(n)]

def topk_idx(scores, k):
    return sorted(range(len(scores)), key=lambda e: scores[e], reverse=True)[:k]

# ---------- Expert:gated FFN ----------
# act=silu -> SwiGLU(Mixtral / Qwen / DeepSeek);act=gelu_tanh -> GeGLU(Gemma 4)
# limit:DeepSeek V4 會 clamp gate(上限)和 up(正負)[src: deepseek_v4 _apply_gate]
class Expert:
    def __init__(self, d_model, d_ff, act=silu, limit=None):
        self.W_gate, self.W_up = rand_matrix(d_model, d_ff), rand_matrix(d_model, d_ff)
        self.W_down = rand_matrix(d_ff, d_model)
        self.act, self.limit = act, limit

    def __call__(self, x):             # x: (n, d_model)
        gate, up = matmul(x, self.W_gate), matmul(x, self.W_up)
        if self.limit is not None:
            L = self.limit
            gate = [[min(g, L) for g in r] for r in gate]
            up = [[max(-L, min(u, L)) for u in r] for r in up]
        h = [[self.act(g) * u for g, u in zip(gr, ur)] for gr, ur in zip(gate, up)]
        return matmul(h, self.W_down)

# 照 expert 分組,每個 expert 只算一次 batch(所有 HF 實作都是這樣)
def run_experts(experts, x, topk):
    out = [[0.0] * len(x[0]) for _ in x]
    for e, expert in enumerate(experts):
        hits = [(t, w) for t, tk in enumerate(topk) for ei, w in tk if ei == e]
        if not hits:
            continue
        y = expert([x[t] for t, _ in hits])
        for (t, w), yr in zip(hits, y):
            out[t] = [o + w * v for o, v in zip(out[t], yr)]
    return out

# ======================================================================
# Routers:都回傳 (scores, topk),topk[t] = [(expert, weight), ...]
# ======================================================================

# 1) Softmax router:Mixtral、Qwen3-MoE、Qwen3.5-MoE [src]
#    softmax -> top-k -> (選擇性)top-k 內重新正規化
#    Mixtral / Qwen3.5-MoE 一定 renorm;Qwen3-MoE 由 config.norm_topk_prob 決定
class SoftmaxRouter:
    balance = "aux_loss"               # 靠 load-balancing loss 平衡
    def __init__(self, d_model, E, k, norm_topk=True):
        self.W, self.k, self.norm = rand_matrix(d_model, E), k, norm_topk

    def __call__(self, x, token_ids=None):
        probs = [softmax(r) for r in matmul(x, self.W)]
        topk = []
        for p in probs:
            idx = topk_idx(p, self.k)
            z = sum(p[e] for e in idx) if self.norm else 1.0
            topk.append([(e, p[e] / z) for e in idx])
        return probs, topk

# 2) Bias router:DeepSeek-V3(sigmoid)、DeepSeek-V4(sqrtsoftplus)[src]
#    - 分數 = score_fn(logit),權重來自「沒加 bias」的分數
#    - 選 expert 時才加 e_score_correction_bias(aux-loss-free 平衡:bias 不走梯度,看 load 調整)
#    - V3 另有 group-limited routing:先選 topk_group 個 group,再在裡面選 top-k
#    - top-k 內 renorm 後乘 routed_scaling_factor
class BiasRouter:
    balance = "bias"
    def __init__(self, d_model, E, k, score=sigmoid, scale=1.0, n_group=1, topk_group=1):
        self.W, self.k, self.score, self.scale = rand_matrix(d_model, E), k, score, scale
        self.n_group, self.topk_group = n_group, topk_group
        self.bias = [0.0] * E

    def choose(self, s):
        c = [v + b for v, b in zip(s, self.bias)]
        if self.n_group > 1:
            g = len(s) // self.n_group
            gs = [sum(sorted(c[i * g:(i + 1) * g], reverse=True)[:2]) for i in range(self.n_group)]
            keep = set(topk_idx(gs, self.topk_group))
            c = [v if e // g in keep else float("-inf") for e, v in enumerate(c)]
        return topk_idx(c, self.k)

    def __call__(self, x, token_ids=None):
        scores = [[self.score(v) for v in r] for r in matmul(x, self.W)]
        topk = []
        for s in scores:
            idx = self.choose(s)
            z = sum(s[e] for e in idx) + 1e-20
            topk.append([(e, s[e] / z * self.scale) for e in idx])
        return scores, topk

    def update_bias(self, load, gamma=0.001):   # [?] 出自 V3 論文:過載的 expert bias 減、閒置的加
        avg = sum(load) / len(load)
        self.bias = [b + gamma * (1 if l < avg else -1) for b, l in zip(self.bias, load)]

# 3) Hash router:DeepSeek-V4 前 3 層 [src: DeepseekV4HashRouter]
#    選哪些 expert 由固定表 tid2eid[token_id] 決定(不學習);權重仍由 gate 分數算
class HashRouter:
    balance = None
    def __init__(self, d_model, E, k, vocab, score=sqrtsoftplus, scale=1.0):
        self.W, self.score, self.scale = rand_matrix(d_model, E), score, scale
        self.tid2eid = [random.sample(range(E), k) for _ in range(vocab)]

    def __call__(self, x, token_ids):
        scores = [[self.score(v) for v in r] for r in matmul(x, self.W)]
        topk = []
        for s, tid in zip(scores, token_ids):
            idx = self.tid2eid[tid]
            z = sum(s[e] for e in idx) + 1e-20
            topk.append([(e, s[e] / z * self.scale) for e in idx])
        return scores, topk

# 4) Gemma 4 router [src: Gemma4TextRouter]
#    先 RMSNorm(不帶 scale)-> 乘可學習 scale 和 d^-0.5 -> Linear -> softmax -> top-k -> renorm
#    最後乘「每個 expert 各一個」的 per_expert_scale(在 renorm 之後,所以權重和不一定是 1)
class GemmaRouter:
    balance = None                     # 讀到的 router / decoder layer 程式裡沒有 aux loss
    def __init__(self, d_model, E, k):
        self.W, self.k, self.d = rand_matrix(d_model, E), k, d_model
        self.scale = [1.0] * d_model
        self.per_expert_scale = [1.0] * E

    def __call__(self, x, token_ids=None):
        h = [[v * s * self.d ** -0.5 for v, s in zip(rms_norm(r), self.scale)] for r in x]
        probs = [softmax(r) for r in matmul(h, self.W)]
        topk = []
        for p in probs:
            idx = topk_idx(p, self.k)
            z = sum(p[e] for e in idx)
            topk.append([(e, p[e] / z * self.per_expert_scale[e]) for e in idx])
        return probs, topk

# ======================================================================
# MoE layer:router + routed experts + (選擇性)shared expert
# ======================================================================
class MoE:
    def __init__(self, router, experts, shared=None, shared_gate=None):
        self.router, self.experts = router, experts
        self.shared = shared           # 一個每個 token 都會過的 Expert
        self.shared_gate = shared_gate # (d_model, 1);有的話 shared 輸出先乘 sigmoid(x @ gate)

    def __call__(self, x, token_ids=None, router_x=None):
        scores, topk = self.router(x if router_x is None else router_x, token_ids)  # Gemma 的 router 吃另一份輸入
        out = run_experts(self.experts, x, topk)
        if self.shared is not None:
            y = self.shared(x)
            if self.shared_gate is not None:                       # Qwen3.5-MoE [src]
                g = [sigmoid(r[0]) for r in matmul(x, self.shared_gate)]
                y = [[g_t * v for v in row] for g_t, row in zip(g, y)]
            out = [[o + v for o, v in zip(orow, yrow)] for orow, yrow in zip(out, y)]

        E = len(self.experts)
        self.load = [0] * E                                         # 每個 expert 分到幾個 token
        for tk in topk:
            for e, _ in tk:
                self.load[e] += 1
        self.aux_loss = (load_balance_loss(scores, topk)
                         if self.router.balance == "aux_loss" else None)
        return out

# Switch Transformer 版 load-balancing loss:E * sum_e f_e * P_e
# f_e = 分給 e 的 token 比例,P_e = router 給 e 的平均機率。完全均勻時 ≈ 1,訓練時乘 ~0.01 加進 loss
def load_balance_loss(probs, topk):
    T, E = len(probs), len(probs[0])
    f = [0.0] * E
    for tk in topk:
        for e, _ in tk:
            f[e] += 1 / (T * len(tk))
    P = [sum(p[e] for p in probs) / T for e in range(E)]
    return E * sum(fe * pe for fe, pe in zip(f, P))

# ======================================================================
# 各 LLM 的版本(E / k 預設值是該模型的規格,測試時會改小)
# ======================================================================

# Mixtral 8x7B:8 expert、top-2、softmax + renorm、沒有 shared expert [src]
def mixtral(d, d_ff, E=8, k=2):
    return MoE(SoftmaxRouter(d, E, k), [Expert(d, d_ff) for _ in range(E)])

# Qwen3-MoE:softmax + (可選)renorm、沒有 shared expert [src]
def qwen3_moe(d, d_ff, E=128, k=8, norm_topk=True):
    return MoE(SoftmaxRouter(d, E, k, norm_topk), [Expert(d, d_ff) for _ in range(E)])

# Qwen3.8-Flash-Next:512 routed、top-10、每層 1 個 shared expert(NVIDIA 文件)
# router 細節 [?]:本機沒有 qwen3_8 原始碼,這裡照最接近的 qwen3_5_moe [src] 寫:
#   softmax + renorm,shared expert 乘 sigmoid gate 後相加
def qwen3_8(d, d_ff, d_shared, E=512, k=10):
    return MoE(SoftmaxRouter(d, E, k), [Expert(d, d_ff) for _ in range(E)],
               shared=Expert(d, d_shared), shared_gate=rand_matrix(d, 1))

# DeepSeek-V3:256 routed、top-8、1 shared、sigmoid、aux-loss-free bias、
# n_group=8 / topk_group=4、routed_scaling_factor=2.5 [src: deepseek_v3 config 預設值]
# shared expert 的中間維度 = d_ff * n_shared
def deepseek_v3(d, d_ff, E=256, k=8, n_group=8, topk_group=4, n_shared=1):
    return MoE(BiasRouter(d, E, k, sigmoid, 2.5, n_group, topk_group),
               [Expert(d, d_ff) for _ in range(E)], shared=Expert(d, d_ff * n_shared))

# DeepSeek-V4(.1):V4.1-Flash model card 寫 384 routed、top-6、1 shared;
# router 照本機 deepseek_v4 [src]:sqrtsoftplus、bias 只用於選擇、renorm 後 * 1.5、
# 前 3 層 hash routing、expert 有 swiglu clamp(limit=10)。V4.1 本身的原始碼本機沒有 [?]
def deepseek_v4(d, d_ff, layer_idx, vocab, E=384, k=6, scale=1.5, num_hash_layers=3):
    if layer_idx < num_hash_layers:
        router = HashRouter(d, E, k, vocab, sqrtsoftplus, scale)
    else:
        router = BiasRouter(d, E, k, sqrtsoftplus, scale)
    return MoE(router, [Expert(d, d_ff, silu, limit=10.0) for _ in range(E)],
               shared=Expert(d, d_ff, silu, limit=10.0))

# Gemma 4 26B-A4B:128 routed、top-8、expert 維度 704、dense MLP 維度 2112 [src + config.json]
# 跟別人最不同:dense MLP 跟 MoE「並聯」(等於 shared expert,但比一般 expert 大),
# 兩邊各自有前後 norm,相加後再過 post_feedforward_layernorm,最後加殘差。
# router 看的是 attention 之後的殘差流(residual),不是 dense MLP 輸入的那份 norm。
class Gemma4FFN:
    def __init__(self, d, d_dense, d_expert, E=128, k=8):
        self.dense = Expert(d, d_dense, gelu_tanh)
        self.moe = MoE(GemmaRouter(d, E, k), [Expert(d, d_expert, gelu_tanh) for _ in range(E)])
        self.ones = [1.0] * d          # 各個 norm 的 gamma,這裡先用 1

    def __call__(self, residual):      # residual: attention 之後的殘差流;回傳已加上殘差的結果
        norm = lambda X: [rms_norm(r, self.ones) for r in X]
        h1 = norm(self.dense(norm(residual)))                      # pre_ff -> dense mlp -> post_ff_1
        h2 = norm(self.moe(norm(residual), router_x=residual))     # pre_ff_2 -> experts -> post_ff_2
        h = norm([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(h1, h2)])   # post_ff
        return [[r + v for r, v in zip(rr, hr)] for rr, hr in zip(residual, h)]

# ---------- 測試 ----------
if __name__ == "__main__":
    random.seed(0)
    T, d, d_ff, vocab = 6, 8, 16, 50
    x = [[random.gauss(0, 1) for _ in range(d)] for _ in range(T)]
    ids = [random.randrange(vocab) for _ in range(T)]

    layers = {
        "mixtral":     mixtral(d, d_ff, E=4, k=2),
        "qwen3_moe":   qwen3_moe(d, d_ff, E=8, k=3),
        "qwen3_8":     qwen3_8(d, d_ff, d_shared=d_ff, E=8, k=3),
        "deepseek_v3": deepseek_v3(d, d_ff, E=8, k=2, n_group=4, topk_group=2),
        "ds_v4 hash":  deepseek_v4(d, d_ff, layer_idx=0, vocab=vocab, E=8, k=2),
        "ds_v4 moe":   deepseek_v4(d, d_ff, layer_idx=5, vocab=vocab, E=8, k=2),
    }
    for name, m in layers.items():
        out = m(x, ids)
        assert len(out) == T and len(out[0]) == d
        w = [round(sum(w for _, w in tk), 3) for tk in m.router(x, ids)[1]][:3]
        print(f"{name:12s} load={m.load} aux={None if m.aux_loss is None else round(m.aux_loss, 3)} weight_sum={w}")

    out = Gemma4FFN(d, d_dense=24, d_expert=8, E=8, k=3)(x)
    assert len(out) == T and len(out[0]) == d
    print("gemma4       ok")

    # aux-loss-free:負載不均時,過載 expert 的 bias 會下降
    r = layers["deepseek_v3"].router
    r.update_bias(layers["deepseek_v3"].load)
    print("bias after update:", [round(b, 4) for b in r.bias])
