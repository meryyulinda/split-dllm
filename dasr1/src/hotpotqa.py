"""HotpotQA (Yang et al., EMNLP 2018), distractor setting: the public long-prompt task.

Every question comes with 10 Wikipedia paragraphs, the 2 that hold the answer and 8
retrieved distractors, all given in the prompt (about 1,000-1,500 tokens), so the prompt
dominates the sequence, as in the authored long-prompt cases, but with public questions
and gold answers. We use the first questions of the official dev set (CC BY-SA 4.0), with the paragraphs
in their official order. The authors' server (curtis.ml.cmu.edu) is not always reachable,
so the data comes from the Hugging Face copy, hotpotqa/hotpot_qa (distractor, validation
= the official dev set, same order).

prepare_from_hub(out_path, n)     writes the first n questions as JSON lines (Hugging Face)
prepare(json_path, out_path, n)   the same from the official hotpot_dev_distractor_v1.json
load_hotpotqa(path, n)            cases in the format of the other workloads
hotpot_score(case, text)          "ok" = the gold answer appears in the reply after the
                                  official answer normalization (lower case, no punctuation
                                  or articles); yes/no questions: the reply's first yes/no.
                                  Also returns the official token F1 of the reply's first line.
"""
from __future__ import annotations

import collections
import json
import re
import string
from pathlib import Path

INSTRUCTION = ("Answer the question using the paragraphs above. Reply with the short answer "
               "only (a name, a number, a few words, or yes or no).")


def prepare(json_path, out_path, n=100):
    rows = json.loads(Path(json_path).read_text(encoding="utf-8"))[: int(n)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("_id", "question", "answer", "type", "level",
                                                  "supporting_facts", "context")}) + "\n")
    return len(rows)


HUB_REPO = "hotpotqa/hotpot_qa"


def prepare_from_hub(out_path, n=100, repo=HUB_REPO):
    """First n questions of the distractor validation split from the Hugging Face copy,
    converted to the official JSON layout (context = [[title, [sentences]], ...])."""
    from huggingface_hub import hf_hub_download, list_repo_files
    try:
        import pandas as pd
        pd.io.parquet.get_engine("auto")
    except ImportError as exc:
        raise SystemExit(f"reading parquet needs pyarrow: pip install pyarrow ({exc})")
    files = []
    for rev in ("main", "refs/convert/parquet"):
        names = list_repo_files(repo, repo_type="dataset", revision=rev)
        files = sorted(f for f in names if f.endswith(".parquet") and "distractor" in f
                       and "validation" in f)
        if files:
            break
    if not files:
        raise SystemExit(f"no distractor validation parquet found in {repo}")
    print(f"HotpotQA from {repo}@{rev}: {files}", flush=True)
    df = pd.concat([pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
                    for f in files], ignore_index=True).head(int(n))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in df.to_dict("records"):
            ctx, sf = r["context"], r["supporting_facts"]
            row = {"_id": r["id"], "question": r["question"], "answer": r["answer"],
                   "type": r["type"], "level": r["level"],
                   "supporting_facts": [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])],
                   "context": [[t, [str(x) for x in sents]]
                               for t, sents in zip(ctx["title"], ctx["sentences"])]}
            f.write(json.dumps(row) + "\n")
    return len(df)


def build_prompt(row):
    paras = [f"Paragraph {i}: {title}\n" + "".join(sents).strip()
             for i, (title, sents) in enumerate(row["context"], start=1)]
    return "\n\n".join(paras) + f"\n\nQuestion: {row['question'].strip()}\n{INSTRUCTION}"


def load_hotpotqa(path, n=None, start=0):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HotpotQA file not found: {path} (run experiments/run_exp23.sh, "
                                "which downloads and prepares it)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[int(start): int(start) + int(n)] if n else rows[int(start):]
    return [{"case_id": f"hotpot_{int(start) + i:04d}", "doc_id": r["_id"], "task": r["type"],
             "question": r["question"], "gold_answer": r["answer"], "prompt": build_prompt(r)}
            for i, r in enumerate(rows)]


# official answer normalization (hotpot_evaluate_v1.py)
def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, gold):
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        if p != g:
            return 0.0
    pt, gt = p.split(), g.split()
    common = collections.Counter(pt) & collections.Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def hotpot_score(case, text):
    reply = str(text).strip()
    first = next((line.strip() for line in reply.splitlines() if line.strip()), "")
    gold = normalize_answer(case["gold_answer"])
    norm = normalize_answer(reply)
    if gold in {"yes", "no"}:
        m = re.search(r"\b(yes|no)\b", norm)
        ok = bool(m) and m.group(1) == gold
    else:
        ok = bool(gold) and f" {gold} " in f" {norm} "
    return {"ok": ok, "extracted": first[:120], "f1": f1_score(first, case["gold_answer"])}
