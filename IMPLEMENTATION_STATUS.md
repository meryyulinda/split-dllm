# Delivered scope and validation status

| Component | Implemented | Verified in this workspace |
| --- | --- | --- |
| Versioned full/zlib/XOR packets; XOR fallback to raw | Yes | NumPy round-trip tests, including BF16 bit patterns and signed zero |
| Sparse native and sparse/full row-wise INT8 | Yes, absolute row updates against reconstructed reference | Repeated reconstruction, omitted-row and checksum tests |
| Stale-reference recovery | Anchor retry when cloud rejects a patch | Loss/forgetting tests; lost response after commitment aborts the query rather than replaying sampling |
| Split denoising, one-request cloud-only, no-network edge-only | Yes | All toy split points/three exact codecs; same generated IDs |
| TCP edge/cloud split | Yes, persistent batch-one RPC | Local loopback test with independent edge/cloud model instances |
| Native LLaDA/Dream partitions | Explicit block adapters; streamed owned safetensors; BF16/FP16/FP32 | Syntax checked; checkpoint/GPU parity must run in Colab |
| FFN selective work | Approximate row execution, dense attention; random active fill plus mandatory changed rows; per-operator audits | Toy row execution and drift tests; real Torch path not executed locally |
| General token-selective attention/KV/Triton kernels | Not implemented | No performance or equivalence claim |
| Table scheduler | Live codec/refresh adaptation at a fixed split; memory/risk filtering and hysteresis | Analytical scheduler and adaptive toy runtime tests |
| Layer-placement adaptation | Offline measured-table trace replay | No live weight migration or physical switch benchmark |
| Author baselines | Thin standalone Fast-dLLM v1 and dLLM-Cache runners | API/source inspected and syntax checked; no pretrained runs |
| Quality/latency CIs | Per-query JSONL and bootstrap intervals; numeric GSM8K pilot scoring | Toy systems pipeline, plotting and CSV export |
| Full RQ5 task suite | JSONL input supports custom prompts; only numeric pilot scorer provided | MATH/MMLU/code/LongBench standardized evaluators remain separate author/harness work |
| Physical wire bytes, energy and FLOPs | Unavailable measurements remain null; analytical link model labeled | No fabricated estimates |
| Colab notebook | Self-contained source bundle, install/preflight/download/parity/sweep/plots/export cells | Structure/code compiled; not remotely executed |

The local environment has Python 3.9, NumPy and Matplotlib, without PyTorch, Transformers or a connected Colab execution tool. No large checkpoint was downloaded locally. Your later Colab execution performs the pretrained and GPU checks. This distinction is recorded in experiment manifests.

The implementation is a runnable research scaffold for RQ1/RQ2 and a restricted RQ3/RQ4 pilot, rather than a completed validated implementation of every proposed paper component. In particular, it must not be called LaCache, Fast-dLLM, d²Cache, or RAC, nor inherit those papers' results.
