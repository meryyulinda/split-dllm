#!/usr/bin/env bash
# One-time setup of a rented edge machine (vast.ai, PyTorch image) for Exp21.
# Run in the home directory of the rented machine:
#   curl -O https://raw.githubusercontent.com/meryyulinda/split-dllm/main/dasr1/experiments/setup_exp21_edge.sh
#   bash setup_exp21_edge.sh
set -euo pipefail
mkdir -p ~/work && cd ~/work
[ -d split-dllm ] || git clone --depth 1 https://github.com/meryyulinda/split-dllm.git
[ -d dLLM-cache-official ] || git clone --depth 1 https://github.com/maomaocun/dLLM-cache.git dLLM-cache-official
[ -d Fast-dLLM ] || git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git
pip install -q "transformers==4.46.2" accelerate sentencepiece pandas safetensors huggingface_hub
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "setup done: cd ~/work/split-dllm/dasr1"
