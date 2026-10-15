# Primary sources screened on 2026-10-12

This is a dated, bounded literature screen, not an exhaustive claim of the latest global SOTA. Repository HEADs can move. The Colab download helper resolves checkpoint commits; external runners record author-repository commits.

| Work | Verified primary source/code | Relevance and comparison scope |
| --- | --- | --- |
| LLaDA | [paper](https://arxiv.org/abs/2502.09992), [official code](https://github.com/lrbob/LLaDA), [checkpoint](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | First deployment target; native masked sampler and bidirectional blocks |
| Dream 7B | [paper](https://arxiv.org/abs/2508.15487), [official code](https://github.com/DreamLM/Dream), [checkpoint](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | Second architecture; preserve its generation conventions |
| Fast-dLLM v1 | [paper](https://arxiv.org/abs/2505.22618), [code](https://github.com/NVlabs/Fast-dLLM/tree/main/v1) | Training-free approximate KV caching and parallel decoding; runner supports standalone prefix cache, with optional sampler change |
| Fast-dLLM v2 | [paper](https://arxiv.org/abs/2509.26328), [repository](https://github.com/NVlabs/Fast-dLLM) | Uses fine-tuning; compare as a different model, not the same-weight deployment baseline |
| dLLM-Cache | [paper](https://arxiv.org/abs/2506.06295), [code](https://github.com/maomaocun/dLLM-cache) | Adaptive feature caching; thin author-code runner, not a reimplementation |
| dKV-Cache | [code and paper links](https://github.com/horseee/dKV-Cache) | Approximate cached-context variants; standalone baseline candidate |
| d²Cache | [paper](https://arxiv.org/abs/2509.23094), [code](https://github.com/Kamichanw/d2Cache) | Token-selective adaptive KV updates; approximate |
| Sparse-dLLM | [paper](https://arxiv.org/abs/2508.02558), [code](https://github.com/OpenMOSS/Sparse-dLLM) | Dynamic cache eviction; candidate for kernel comparison |
| Elastic-Cache | [paper](https://arxiv.org/abs/2510.14973), [code](https://github.com/VILA-Lab/elastic-cache) | Attention-aware refresh and depth adaptation; code reports corrected throughput figures, so use a recorded revision |
| Dynamic-dLLM | [paper](https://arxiv.org/abs/2606.26120), [code](https://github.com/TianyiWu233/DYNAMIC-DLLM) | Cache-budget adaptation plus parallel decoding; separate the two factors |
| Fast-dLLM++ | [paper](https://arxiv.org/abs/2606.02955), [author-linked code](https://github.com/Ringo-Star/FastdLLM_plusplus) | Commit-set selection; secondary sampler experiment |
| LaCache | [paper and detailed method](https://arxiv.org/html/2607.16339v1) | Exact first-layer/operator memoization is distinct from its FP8 component; a verified executable author repository was not established by this screen |
| RAC | [paper](https://arxiv.org/abs/2608.04991) | Strong reference-aware codec comparison; its local-cloud-local causal topology differs from the one-boundary dLLM experiment |
| Sangam | [paper](https://arxiv.org/abs/2607.04206), [code](https://github.com/UT-InfraAI/sangam) | Serving/scheduling for cached dLLMs; does not by itself establish a layer-split edge-cloud codec |
| Edge DLM survey | [paper](https://arxiv.org/abs/2609.04778) | Recent edge/cloud deployment context and open problems; survey, not a measured split-runtime competitor |
| Edge block-diffusion hardware | [paper](https://arxiv.org/abs/2609.01084) | Recent edge hardware/compression co-design; modeled platforms and native block diffusion must be distinguished from unmodified full-attention LLaDA |

Implementation source inspection also covered the checkpoints' `modeling_llada.py`, `modeling_dream.py` and Dream `generation_utils.py`, plus official LLaDA `generate.py`. The adapters call native model blocks. The samplers here implement a restricted deterministic configuration and must pass the provided author-code parity gate. No third-party source file was vendored into this package.

Sources with overlapping method names must be checked by title and identifier. For example, the ICML 2025 ladder-shaped causal KV cache named LaCache is a different work from arXiv:2607.16339.
