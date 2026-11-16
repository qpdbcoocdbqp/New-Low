# References: [MoE](moe.py)

<!-- Marker: [paper] = paper; title and first author checked against the arxiv.org/abs page on 2026-11-16. -->

## Core Concepts

### Sparsely-Gated MoE / top-k routing
- [paper] Shazeer et al., 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. https://arxiv.org/abs/1701.06538

### Load-balancing loss
- [paper] Fedus et al., 2021. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. https://arxiv.org/abs/2101.03961

### Auxiliary-loss-free load balancing
- [paper] Wang et al., 2024. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*. https://arxiv.org/abs/2408.15664

### Hash routing
- [paper] Roller et al., 2021. *Hash Layers For Large Sparse Models*. https://arxiv.org/abs/2106.04426

### SwiGLU / GeGLU
- [paper] Shazeer, 2020. *GLU Variants Improve Transformer*. https://arxiv.org/abs/2002.05202

### SiLU
- [paper] Elfwing et al., 2017. *Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*. https://arxiv.org/abs/1702.03118

### GELU
- [paper] Hendrycks & Gimpel, 2016. *Gaussian Error Linear Units (GELUs)*. https://arxiv.org/abs/1606.08415

### RMSNorm
- [paper] Zhang & Sennrich, 2019. *Root Mean Square Layer Normalization*. https://arxiv.org/abs/1910.07467

## Models

### Mixtral
- [paper] Jiang et al., 2024. *Mixtral of Experts*. https://arxiv.org/abs/2401.04088

### Qwen3-MoE
- [paper] Yang et al., 2025. *Qwen3 Technical Report*. https://arxiv.org/abs/2505.09388

### Qwen3.8-Flash-Next
- [paper] Qwen Team, 2026. *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*. https://arxiv.org/abs/2608.30320

### DeepSeek-V3
- [paper] Dai et al., 2024. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. https://arxiv.org/abs/2401.06066
- [paper] DeepSeek-AI, 2024. *DeepSeek-V3 Technical Report*. https://arxiv.org/abs/2412.19437

### DeepSeek-V4
- [paper] DeepSeek-AI, 2026. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. https://arxiv.org/abs/2606.19348

### Gemma 4
- [paper] Gemma Team, 2026. *Gemma 4 Technical Report*. https://arxiv.org/abs/2607.02770

## Further Reading

- [paper] Lepikhin et al., 2020. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. https://arxiv.org/abs/2006.16668
- [paper] Zhou et al., 2022. *Mixture-of-Experts with Expert Choice Routing*. https://arxiv.org/abs/2202.09368
