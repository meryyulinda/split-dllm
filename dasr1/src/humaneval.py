"""HumanEval (Chen et al., 2021) for Exp20: problems, prompt, code extraction, execution.

Data: data/HumanEval.jsonl.gz of https://github.com/openai/human-eval (cloned next to
split-dllm by run_exp20.sh). Scoring is pass@1 with one sample: the generated function
is executed against the problem's unit tests in a separate Python process with a
time limit and a memory limit. The code comes from the model, so this runs untrusted
code on the server, as every HumanEval evaluation does; the child process gets a
temporary working directory, -I (isolated mode), no stdin, 10 s and 2 GiB.
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

INSTRUCTION = ("Complete the following Python function. Reply with the complete function "
               "in one Python code block.")


def load_humaneval(path, n=None):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[:n] if n else rows
    return [{"case_id": r["task_id"].replace("/", "_"), "task": "code",
             "prompt": f"{INSTRUCTION}\n\n```python\n{r['prompt']}```",
             "he_prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"],
             "gold_answer": ""} for r in rows]


_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text, case):
    """Program to test: the function from the reply, with the prompt's imports."""
    blocks = _BLOCK.findall(text)
    if not blocks and "```" in text:          # unterminated block (answer cut off)
        blocks = [text.split("```", 1)[1].split("\n", 1)[-1]]
    entry = case["entry_point"]
    code = next((b for b in blocks if f"def {entry}" in b), blocks[0] if blocks else text)
    prompt = case["he_prompt"]
    imports = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
    if f"def {entry}" in code:
        return f"{imports}\n{code}"
    return prompt + code                       # reply is only the function body


def _limits():  # pragma: no cover - runs in the child
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def run_tests(program, case, timeout_s=10.0):
    src = f"{program}\n\n{case['test']}\n\ncheck({case['entry_point']})\n"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", src], cwd=tmp, stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=timeout_s, preexec_fn=_limits)
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def humaneval_score(case, text):
    program = extract_code(text, case)
    return {"ok": run_tests(program, case), "extracted": None}
