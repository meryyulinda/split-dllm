"""Regenerate the active notebook with a self-contained Colab source snapshot."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import textwrap
import zipfile


def main():
    package = Path(__file__).resolve().parents[1]
    workspace = package.parent
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            relative = path.relative_to(workspace)
            if (path.is_file() and path.suffix in (".py", ".md", ".txt", ".json", ".tex")
                    and "results" not in path.parts and "__pycache__" not in path.parts):
                archive.write(path, relative.as_posix())
        entry_point = workspace / "run_experiments.py"
        if entry_point.is_file():
            archive.write(entry_point, entry_point.name)
    payload = base64.b64encode(buffer.getvalue()).decode()
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    cells = []
    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=textwrap.dedent(text).strip().splitlines(keepends=True)))
    def code(text):
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                          source=textwrap.dedent(text).strip().splitlines(keepends=True)))
    md("""
    # Diffusion-Aware Split Runtime — LLaDA experiments in Colab

    Start with **Runtime → Change runtime type → GPU**. This notebook embeds the experiment source, so no repository upload is needed. Run cells in order. The first model download is roughly the native 8B weight size and takes time.

    One Colab GPU represents the cloud. CPU prefix execution represents an **emulated edge**, not an RTX 3060 or a physical network. Application byte measurements and analytical bandwidth/RTT estimates are labeled separately from measured compute latency. Energy and FLOPs remain unavailable rather than being invented.

    The main pilot holds weights and deterministic sampling fixed. It tests full, compressed, XOR and approximate sparse/INT8 boundary transmission. Optional FFN-only caching skips row work but leaves attention dense. General sparse attention and live layer migration are not implemented.

    The package includes a detailed paper review and a dated primary-source screen (2026-10-12). Pretrained GPU measurements have **not** already been run by the authoring agent.
    """)
    code(f"""
    import base64, hashlib, io, os, sys, zipfile
    from pathlib import Path
    PROJECT = Path('/content/dasr_project') if Path('/content').exists() else Path.cwd() / 'dasr_colab_workspace'
    PROJECT.mkdir(parents=True, exist_ok=True)
    SOURCE_ARCHIVE_B64 = {payload!r}
    blob = base64.b64decode(SOURCE_ARCHIVE_B64)
    assert hashlib.sha256(blob).hexdigest() == {digest!r}
    with zipfile.ZipFile(io.BytesIO(blob)) as bundle:
        for member in bundle.infolist():
            target = (PROJECT / member.filename).resolve()
            assert PROJECT.resolve() in target.parents, 'Archive path escaped project'
        bundle.extractall(PROJECT)
    os.chdir(PROJECT)
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    print('Source extracted to', PROJECT)
    """)
    md("""
    Install the pinned compatibility environment while retaining Colab's CUDA PyTorch. If Colab reports imported package changes, restart the session and rerun extraction before continuing. Dependencies are a conservative starting point; the parity gate below determines whether this checkpoint/environment is valid.
    """)
    code("""
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'dasr/requirements-colab.txt'], check=True)
    import torch
    print('PyTorch:', torch.__version__)
    assert tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 5), 'Use torch >= 2.5'
    """)
    md("""
    Protocol tests run without large weights. They check exact reconstruction, stale references, approximate reference synchronization, bidirectional drift, selective FFN row work, fixed-split adaptation and real loopback TCP.
    """)
    code("""
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'dasr/tests', '-v'], check=True)
    from dasr.scripts.colab_support import hardware
    HW = hardware()
    print(HW)
    """)
    md("""
    Resolve the exact checkpoint revision and inspect candidate placement memory before downloading weights. A T4 uses FP16 consistently; newer GPUs use BF16. The model remains native precision for that dtype. The sizing guard reserves GPU memory for logits/softmax and host memory for streaming and codecs. It excludes extra FFN-cache storage; that optional experiment needs more headroom.
    """)
    code("""
    from huggingface_hub import HfApi
    from dasr.scripts.colab_support import choose_split
    MODEL_ID = 'GSAI-ML/LLaDA-8B-Instruct'
    MODEL_REVISION = HfApi().model_info(MODEL_ID).sha
    DTYPE = HW['dtype']
    GEN_LENGTH, STEPS, BLOCK_LENGTH = 32, 16, 32  # Functional pilot; expand for benchmark quality.
    _, PLAN, ALL_PLANS = choose_split(MODEL_ID, MODEL_REVISION, DTYPE, GEN_LENGTH, prompt_tokens=256)
    SPLIT = PLAN['split']
    print('Pinned revision:', MODEL_REVISION)
    print('Selected CPU-edge/GPU-cloud plan:', PLAN)
    print('Other plans:', ALL_PLANS)
    """)
    md("""
    Download the pinned pretrained checkpoint. This cell is the later download requested in the chat. It downloads actual weights only when you execute it in Colab. The official LLaDA repository is also fetched for sampler validation; its commit is recorded.
    """)
    code("""
    import json
    from huggingface_hub import snapshot_download
    CHECKPOINT = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=['*.json', '*.py', '*.safetensors', '*.model', '*.txt'])
    OFFICIAL = PROJECT / 'author_code' / 'LLaDA'
    OFFICIAL.parent.mkdir(parents=True, exist_ok=True)
    if not OFFICIAL.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/lrbob/LLaDA.git', str(OFFICIAL)], check=True)
    OFFICIAL_COMMIT = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    from dasr.metrics import write_json
    write_json('results/download_manifest.json', {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
        'official_repository_commit': OFFICIAL_COMMIT, 'checkpoint': CHECKPOINT, 'hardware': HW})
    print(CHECKPOINT)
    """)
    md("""
    The native-model gate compares unsplit author forward against explicit partition forwards and compares the deterministic sampler against author code. It requires the full model on one device. Use a large GPU when it fits, or a high-RAM CPU runtime; CPU validation can be slow. Insufficient memory is an infeasible configuration, not permission to silently quantize the reference.

    If the gate cannot run on this Colab allocation, the functional pilot remains usable, but its outputs must not be presented as validated exact-tier paper results. To validate later, run the same command on a sufficiently provisioned machine with the same checkpoint and dtype.
    """)
    code("""
    import psutil
    total_weights = PLAN['edge_weight_gib'] + PLAN['cloud_weight_gib']
    VALIDATION_DEVICE = ('cuda:0' if total_weights + 3 < HW['gpu_free_gib'] else
                         'cpu' if total_weights + 3 < psutil.virtual_memory().available / 2**30 else None)
    VALIDATION_REPORT = PROJECT / 'results/native_validation.json'
    if VALIDATION_DEVICE is None:
        print('Native full-model parity is infeasible in this allocation. Pilot results will stay unvalidated.')
        VALIDATION_REPORT = None
    else:
        command = [sys.executable, '-m', 'dasr', 'validate-model', '--backend', 'llada',
            '--checkpoint', CHECKPOINT, '--revision', MODEL_REVISION, '--dtype', DTYPE,
            '--edge-device', VALIDATION_DEVICE, '--cloud-device', VALIDATION_DEVICE,
            '--gen-length', '8', '--steps', '4', '--block-length', '8',
            '--official-repo', str(OFFICIAL), '--output', str(VALIDATION_REPORT)]
        subprocess.run(command, check=True)
        print(VALIDATION_REPORT.read_text())
    """)
    md("""
    Prepare 30 fixed zero-shot GSM8K questions, with dataset revision and indices saved. This is a reproducible numeric pilot, not the official few-shot evaluation protocol. Keep a different set of IDs for calibration. Thirty examples can establish a systems pilot; they usually cannot establish small quality losses. Short 32-token answers will often truncate reasoning, so use 256/512 tokens for a benchmark-quality study after checking memory and runtime.
    """)
    code("""
    DATASET_REVISION = HfApi().dataset_info('openai/gsm8k').sha
    subprocess.run([sys.executable, '-m', 'dasr', 'prepare-gsm8k', '--limit', '30',
        '--seed', '7', '--revision', DATASET_REVISION, '--output', 'data/gsm8k_pilot.jsonl'], check=True)
    print('Dataset revision:', DATASET_REVISION)
    """)
    md("""
    Run a 3-query functional sweep first. The checkpoint is loaded once for all codecs, with CPU edge weights and GPU cloud weights streamed directly into place. The measured latency includes tensor copies, codec work and serialization. The reported modeled latency adds the stated 100 Mbit/s / 20 ms analytical link cost. No real network is claimed.
    """)
    code("""
    COMMON = [sys.executable, '-m', 'dasr', 'bench', '--backend', 'llada', '--checkpoint', CHECKPOINT,
        '--revision', MODEL_REVISION, '--dtype', DTYPE, '--split', str(SPLIT),
        '--edge-device', 'cpu', '--cloud-device', 'cuda:0', '--data', 'data/gsm8k_pilot.jsonl',
        '--gen-length', str(GEN_LENGTH), '--steps', str(STEPS), '--block-length', str(BLOCK_LENGTH),
        '--warmup', '1', '--bandwidth-mbps', '100', '--rtt-ms', '20']
    if VALIDATION_REPORT is not None:
        COMMON += ['--validation-report', str(VALIDATION_REPORT)]
    subprocess.run(COMMON + ['--limit', '3', '--sweep', 'dasr/configs/colab_pilot.json',
                            '--output', 'results/llada_smoke'], check=True)
    subprocess.run([sys.executable, '-m', 'dasr', 'analyze', '--input', 'results/llada_smoke'], check=True)
    """)
    code("""
    import pandas as pd
    from IPython.display import display, Image
    display(pd.read_csv('results/llada_smoke/summary.csv'))
    display(Image(filename='results/llada_smoke/tradeoffs.png'))
    print(json.loads(Path('results/llada_smoke/manifest.json').read_text()))
    """)
    md("""
    The following larger runs are optional and can consume substantial Colab time. Toggle the corresponding variable after the smoke run succeeds. For quality experiments, update lengths in COMMON first and use standardized author/harness prompts. The pilot scores and plots do not replace HumanEval execution, MMLU scoring, MATH equivalence or LongBench metrics.
    """)
    code("""
    RUN_30_QUERY_PILOT = False
    if RUN_30_QUERY_PILOT:
        subprocess.run(COMMON + ['--limit', '30', '--sweep', 'dasr/configs/colab_pilot.json',
                                '--output', 'results/llada_pilot'], check=True)
        subprocess.run([sys.executable, '-m', 'dasr', 'analyze', '--input', 'results/llada_pilot'], check=True)
    """)
    md("""
    RQ1 instrumentation runs separately from latency runs. This cell logs per-row boundary drift and plots the first query's trace. Repeat at several memory-feasible splits for the layer dimension. The FFN experiment is a separate optional approximate ablation and can need significant additional GPU/CPU cache memory.
    """)
    code("""
    RUN_DRIFT_PROFILE = False
    RUN_FFN_ABLATION = False
    if RUN_DRIFT_PROFILE:
        subprocess.run(COMMON + ['--limit', '3', '--codec', 'full', '--trace-drift',
                                '--output', 'results/llada_drift'], check=True)
        subprocess.run([sys.executable, '-m', 'dasr', 'analyze', '--input', 'results/llada_drift'], check=True)
    if RUN_FFN_ABLATION:
        subprocess.run(COMMON + ['--limit', '3', '--sweep', 'dasr/configs/ffn_ablation.json',
                                '--output', 'results/llada_ffn'], check=True)
        subprocess.run([sys.executable, '-m', 'dasr', 'analyze', '--input', 'results/llada_ffn'], check=True)
    """)
    md("""
    Expand network conditions from measured compute logs. This produces labeled analytical estimates, not packet-capture measurements. Live adaptation needs a measured calibration profile and held-out evaluation IDs. See README commands for profile-table, fixed-split adaptive codecs, and placement replay.
    """)
    code("""
    subprocess.run([sys.executable, '-m', 'dasr.scripts.network_sweep', '--input',
        'results/llada_smoke/xor_dense.jsonl', '--output', 'results/network_estimates.csv'], check=True)
    display(pd.read_csv('results/network_estimates.csv').head())
    """)
    md("""
    Author baselines run in separate processes/environments using `dasr/scripts/external_baselines.py`. Use Fast-dLLM **v1** for training-free comparisons; the v2 checkpoint is a different trained model. dLLM-Cache requires its author hooks. Neither standalone author-cache baseline is automatically composed with the split runtime. A full GPU baseline must fit or be reported infeasible. README includes exact invocations and scope.

    For Dream, use its own checkpoint, `--backend dream`, full-canvas block length and `validate-model`; the provided adapter includes native position/logit conventions. RQ5 still requires actual two-model/multiple-task measurements.
    """)
    code("""
    from IPython.display import Markdown
    display(Markdown(Path('dasr/PAPER_REVIEW.md').read_text()))
    """)
    md("""
    Export scripts, logs and manifests before Colab disconnects. The archive excludes downloaded model weights and author repositories. Download it using the final optional line, or copy it to your own persistent storage.
    """)
    code("""
    import zipfile
    OUTPUT_ARCHIVE = PROJECT / 'dasr_experiment_artifacts.zip'
    with zipfile.ZipFile(OUTPUT_ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for directory in ('dasr', 'results', 'data'):
            for path in (PROJECT / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    bundle.write(path, path.relative_to(PROJECT))
    print(OUTPUT_ARCHIVE)
    # from google.colab import files
    # files.download(str(OUTPUT_ARCHIVE))
    """)
    for index, cell in enumerate(cells):
        cell["id"] = "dasr-%02d" % index
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                    language_info=dict(name="python"), colab=dict(name="Untitled-1.ipynb", provenance=[]),
                    accelerator="GPU", dasr=dict(source_sha256=digest)), nbformat=4, nbformat_minor=5)
    target = workspace / "Untitled-1.ipynb"
    backup = workspace / "Untitled-1.original.ipynb"
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print("Created", target, "with", len(cells), "cells; source archive SHA256", digest)


if __name__ == "__main__":
    main()
