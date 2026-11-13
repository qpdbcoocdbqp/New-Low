"""
GRPO (Group Relative Policy Optimization) in pure Python -- no numpy / torch.

玩具設定:
  - "LLM"      : 自回歸策略 π(token | position, prev_token)，參數是 logits 表
  - "prompt"   : 固定 (這裡省略，等於單一 prompt)
  - "verifier" : 可驗證獎勵 (RLVR)，序列總和 == TARGET 得 1，否則 0
  - 對每個 prompt 取樣 G 個 completion 當一組，用組內相對分數當 advantage，不需要 critic

核心公式 (token level):
  A_i      = (r_i - mean(r)) / (std(r) + eps)                  # 組內標準化
  ratio    = exp(logp_new - logp_old)
  L_policy = -min(ratio*A, clip(ratio, 1-eps_lo, 1+eps_hi)*A)  # PPO-clip
  L_kl     = beta * (exp(ref-logp) - (ref-logp) - 1)           # k3 estimator, 對 ref model
  loss     = mean_over_group( mean_over_tokens( L_policy + L_kl ) )

變體開關:
  - dr_grpo  : 不除 std、token 加總除以固定長度 (Dr.GRPO，去除長度/難度偏差)
  - clip_high: eps_hi > eps_lo (DAPO 的 clip-higher，避免熵崩塌)
  - beta=0   : 不用 KL / 不需要 ref model (DAPO、近期多數 RLVR 做法)
"""
import math
import random

VOCAB = 5          # token 0..4
SEQ_LEN = 4
TARGET = 10        # 要求 sum(tokens) == TARGET


# ---------------------------------------------------------------- policy
class Policy:
    """logits[(pos, prev)] -> list[VOCAB]。prev=-1 代表 BOS。"""

    def __init__(self):
        self.logits = {
            (pos, prev): [0.0] * VOCAB
            for pos in range(SEQ_LEN)
            for prev in range(-1, VOCAB)
        }

    def copy(self):
        p = Policy()
        p.logits = {k: v[:] for k, v in self.logits.items()}
        return p

    @staticmethod
    def softmax(zs):
        m = max(zs)
        es = [math.exp(z - m) for z in zs]
        t = sum(es)
        return [e / t for e in es]

    def probs(self, pos, prev):
        return self.softmax(self.logits[(pos, prev)])

    def sample(self):
        toks, prev = [], -1
        for pos in range(SEQ_LEN):
            p = self.probs(pos, prev)
            t = random.choices(range(VOCAB), weights=p)[0]
            toks.append(t)
            prev = t
        return toks

    def token_logps(self, toks):
        out, prev = [], -1
        for pos, t in enumerate(toks):
            out.append(math.log(self.probs(pos, prev)[t]))
            prev = t
        return out


def reward_fn(toks):
    return 1.0 if sum(toks) == TARGET else 0.0


# ---------------------------------------------------------------- GRPO
def grpo_step(policy, ref, *, group_size=32, n_prompts=1, mu=4, lr=0.5,
              eps_lo=0.2, eps_hi=0.2, beta=0.02, dr_grpo=False):
    """一次 rollout (G 個樣本) + mu 次梯度更新 (同一批資料重複使用，ratio/clip 才有意義)。"""
    old = policy.copy()  # π_old: 取樣用的策略，在 mu 次更新中固定

    # 1. rollout + reward + group-relative advantage
    group = []
    for _ in range(group_size):
        toks = old.sample()
        group.append((toks, reward_fn(toks)))
    rs = [r for _, r in group]
    mean = sum(rs) / len(rs)
    std = math.sqrt(sum((r - mean) ** 2 for r in rs) / len(rs))
    if std < 1e-8 and not dr_grpo:
        return mean, 0.0  # 全對/全錯 -> advantage 全 0，沒有學習訊號 (DAPO 會做 dynamic sampling 丟棄)
    advs = [(r - mean) if dr_grpo else (r - mean) / (std + 1e-4) for r in rs]

    batch = []
    for (toks, _), a in zip(group, advs):
        batch.append((toks, a, old.token_logps(toks), ref.token_logps(toks)))

    # 2. mu 次 PPO-style 更新
    for _ in range(mu):
        grads = {k: [0.0] * VOCAB for k in policy.logits}
        for toks, A, logp_old, logp_ref in batch:
            # 序列長度固定，所以 GRPO 的 1/|o| 與 Dr.GRPO 的固定常數相同；
            # 變長輸出時 GRPO 要除以各自的 len(toks)，Dr.GRPO 則除以固定的 max_len。
            norm = 1.0 / SEQ_LEN
            prev = -1
            for pos, t in enumerate(toks):
                p = policy.probs(pos, prev)
                logp = math.log(p[t])
                ratio = math.exp(logp - logp_old[pos])

                # d(L_policy)/d(logp): 被 clip 到的 token 梯度為 0
                active = (A > 0 and ratio < 1 + eps_hi) or (A < 0 and ratio > 1 - eps_lo)
                dpol = -A * ratio if active else 0.0
                # d(L_kl)/d(logp), k3 = exp(ref-logp) - (ref-logp) - 1
                dkl = beta * (1.0 - math.exp(logp_ref[pos] - logp))
                coef = (dpol + dkl) * norm / len(batch)

                # d logp / d logits = onehot - p
                g = grads[(pos, prev)]
                for k in range(VOCAB):
                    g[k] += coef * ((1.0 if k == t else 0.0) - p[k])
                prev = t
        for key, g in grads.items():  # SGD
            z = policy.logits[key]
            for k in range(VOCAB):
                z[k] -= lr * g[k]
    return mean, std


# ---------------------------------------------------------------- eval / main
def evaluate(policy, n=2000):
    return sum(reward_fn(policy.sample()) for _ in range(n)) / n


def train(name, steps=150, **kw):
    random.seed(0)
    policy = Policy()
    ref = policy.copy()  # 凍結的 reference model (SFT 模型)
    print(f"== {name} ==  init acc={evaluate(policy):.3f}")
    for step in range(1, steps + 1):
        mean_r, _ = grpo_step(policy, ref, **kw)
        if step % 30 == 0:
            print(f"step {step:4d}  rollout mean reward={mean_r:.3f}  eval acc={evaluate(policy):.3f}")
    print("sample:", [policy.sample() for _ in range(3)], "\n")


if __name__ == "__main__":
    train("GRPO (beta=0.02)")
    train("DAPO-style (clip-higher, beta=0)", eps_hi=0.28, beta=0.0)
    train("Dr.GRPO (no std norm, 較大 lr 補償 advantage 縮小)", dr_grpo=True, lr=1.5)
