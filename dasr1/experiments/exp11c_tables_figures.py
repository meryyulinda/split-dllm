"""Exp11c: paper tables and figures from the Exp11 runs (no GPU needed).

Run from dasr1/:  python experiments/exp11c_tables_figures.py

Reads the Exp11 output folders (GSM8K and authored long prompts), re-runs the
Exp11b network emulation, and writes to outputs/exp11c/:

  flops_per_run.csv       analytic edge / cloud FLOPs and row work per run
  main_table.csv/.tex     Table "main results" (Dream rows)
  tradeoff_table.csv/.tex Table "quality-systems trade-off" (GSM8K)
  fig_latency_bandwidth.pdf   latency vs bandwidth, both workloads
  fig_tradeoff_gsm8k.pdf      accuracy vs latency and vs uplink (GSM8K)
  fig_uplink_timeline.pdf     uplink bytes per denoising step, one prompt each
  emulated_summary_<folder>.csv   full latency grid per workload (latency table)

Data source (what the paper reports): the dedicated-GPU session written by
run_exp11_clean.sh (EXP11C_WORKLOADS, default *_addon_clean / *_clean folders).
Every timing and energy number comes from that session. The Dream clean runs do
not contain C1 (feature cache, full boundary): C1 performs exactly the computation
of C2 and only sends every row, so it is derived from C2's steps with full-boundary
bytes (C2 reproduced C1 step by step on every prompt of the original session).
If the earlier sessions' folders exist (EXP11C_CHECK_DIRS), the script checks that
answers and byte counts are identical across sessions and that the derived C1
bytes equal the measured ones.
Optional add-on folders (EXP11C_ADDON_DIRS, default none) can still contribute
variants that a workload folder lacks.

FLOPs model (multiply-adds x 2, Dream-v0-Instruct-7B: d=3584, MLP 18944,
28 query / 4 KV heads of 128, vocabulary 152064, 28 layers). Per layer and
step, depending on the refresh decision of the feature cache:
  full refresh          all n rows: QKV, attention n x n, O, MLP
  answer refresh        answer rows G: QKV, attention G x n, O, MLP
  prompt refresh only   prompt rows + floor(rho*G) rows: QK, V for all G,
                        attention, O, MLP
  no refresh            V for all G; floor(rho*G) rows: QK, attention, O, MLP
Layer 1 is always a full refresh. The LM head runs on all n rows every step
(Dream returns logits for every position). Embedding lookups, norms, softmax
and the cosine-similarity selection are ignored (< 1%).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp11b_network_emulation as emu  # noqa: E402

try:
    import configs.experiment_config as cfg
except Exception:  # pragma: no cover
    cfg = None


def setting(name, default):
    return getattr(cfg, name, default) if cfg is not None else default


# Model dimensions for the analytic FLOPs. beta = prompt-tail rows moved to the
# answer cache (Dream's logit shift); LLaDA has none.
ARCH = {
    "dream": dict(D=3584, D_KV=512, D_FF=18944, VOCAB=152064, N_LAYERS=28, beta=1),
    "llada": dict(D=4096, D_KV=4096, D_FF=12288, VOCAB=126464, N_LAYERS=32, beta=0),
}
SPLIT = int(setting("EXP11_SPLIT_LAYER", 4))
RHO = 0.25
UP_HEADER_B = 16  # uplink packet header, as in src/split_runtime.py
# label -> (folder, answer tokens, model); Dream first (figures use Dream).
# Default = the dedicated-GPU session (run_exp11_clean.sh); a workload whose folder
# does not exist is skipped.
WORKLOADS = {label: spec for label, spec in dict(setting("EXP11C_WORKLOADS", {
    "GSM8K": ("exp11_gsm8k_k4_t256_T0.1_addon_clean", 256, "dream"),
    "Long prompts": ("exp11_authored_k4_t128_T0.1_addon_clean", 128, "dream"),
    "GSM8K (LLaDA)": ("exp11_llada_gsm8k_k4_t256_T0_clean", 256, "llada"),
})).items() if (ROOT / "outputs" / spec[0] / "runs.csv").exists()}
DREAM_LABELS = [k for k, v in WORKLOADS.items() if v[2] == "dream"]
# Earlier sessions (shared GPUs): only used to check that outputs and bytes match.
CHECK_DIRS = dict(setting("EXP11C_CHECK_DIRS", {
    "GSM8K": ["exp11_gsm8k_k4_t256_T0.1", "exp11_gsm8k_k4_t256_T0.1_addon"],
    "Long prompts": ["exp11_authored_k4_t128_T0.1", "exp11_authored_k4_t128_T0.1_addon"],
    "GSM8K (LLaDA)": ["exp11_llada_gsm8k_k4_t256_T0"],
}))
ADDON_DIRS = dict(setting("EXP11C_ADDON_DIRS", {}))
DISPLAY = {"GSM8K": "Dream, GSM8K", "Long prompts": "Dream, long prompts",
           "GSM8K (LLaDA)": "LLaDA, GSM8K"}
REF_NET = dict(bandwidth_mbps=100, rtt_ms=20, edge_slowdown=3.0)
OUT = ROOT / "outputs" / "exp11c"

NAMES = {
    "cloud_only": "Cloud-only",
    "cloud_only_cache": "Cloud-only + feature cache",
    "S0_split_full": "Static split / full",
    "S1_split_exact_delta": "Static split / exact delta",
    "S2_split_int8": "Static split / compressed (INT8)",
    "C1_cache_full": "Feature cache, full boundary",
    "C2_cache_aware": "DASR exact (cache-aware)",
    "C3_cache_aware_exact": "DASR exact + exact delta",
    "C4_cache_aware_int8": "DASR approximate (INT8)",
    "A1_adaptive_cache_aware": "DASR adaptive",
}


# ----------------------------------------------------------------- FLOPs
def layer_flops(q_rows, kv_rows, v_only_rows, attn_q, n, a):
    """FLOPs of one decoder layer for one step (a = ARCH entry)."""
    D, D_KV, D_FF = a["D"], a["D_KV"], a["D_FF"]
    qo = 2 * 2 * q_rows * D * D                    # Q and O projections
    kv = 2 * 2 * kv_rows * D * D_KV                # K and V projections
    v_extra = 2 * v_only_rows * D * D_KV           # V-only projections (selection)
    attn = 4 * attn_q * n * D                      # QK^T and AV
    mlp = 6 * q_rows * D * D_FF                    # gate, up, down
    return qo + kv + v_extra + attn + mlp


def step_layer_flops(n, P, G, gen_refresh, prompt_refresh, cached, a):
    if not cached or (gen_refresh and prompt_refresh):
        return layer_flops(n, n, 0, n, n, a)
    U = int(G * RHO)
    if gen_refresh:
        return layer_flops(G, G, 0, G, n, a)
    if prompt_refresh:
        return layer_flops(P + U, P + U, G - U, P + U, n, a)
    return layer_flops(U, U, G - U, U, n, a)


def run_flops(steps, prompt_tokens, answer_tokens, cached, arch="dream"):
    """Edge / cloud FLOPs and row work of one run (steps: that run's steps.csv rows)."""
    a = ARCH[arch]
    beta = a["beta"] if cached else 0
    n = prompt_tokens + answer_tokens
    P, G = prompt_tokens - beta, answer_tokens + beta
    edge = cloud = work = 0.0
    for _, s in steps.iterrows():
        g = bool(s.get("gen_refresh", True)) if cached else True
        p = bool(s.get("prompt_refresh", True)) if cached else True
        per_layer = step_layer_flops(n, P, G, g, p, cached, a)
        full = layer_flops(n, n, 0, n, n, a)
        edge += full + (SPLIT - 1) * per_layer            # layer 1 always full
        cloud += (a["N_LAYERS"] - SPLIT) * per_layer + 2 * n * a["D"] * a["VOCAB"]
        rows = n if not cached else ((P if p else 0) + (G if g else RHO * G))
        work += rows / n
    return edge, cloud, work / len(steps)


def to_bool(series):
    return series.map(lambda v: str(v).strip().lower() == "true")


def derive_c1(label, runs, steps):
    """C1 = C2's computation with the full boundary sent every step (see docstring)."""
    if "C1_cache_full" in set(runs.variant) or "C2_cache_aware" not in set(runs.variant):
        return runs, steps
    c1s = steps[steps.variant == "C2_cache_aware"].copy()
    c1s["variant"] = "C1_cache_full"
    c1s["up_bytes"] = c1s["full_bytes"] + UP_HEADER_B
    c1s["prompt_sent"] = True
    c1r = runs[runs.variant == "C2_cache_aware"].copy()
    c1r["variant"] = "C1_cache_full"
    c1r["up_mb"] = (c1s.groupby("case_id").up_bytes.sum() / 1e6).reindex(c1r.case_id).values
    for col in ("edge_peak_gib", "cloud_peak_gib", "edge_energy_j", "cloud_energy_j",
                "edge_active_j", "cloud_active_j", "wall_s"):
        if col in c1r:
            c1r[col] = np.nan  # not measured for C1
    print(f"[{label}] C1 derived from C2 (same computation, full boundary)", flush=True)
    return (pd.concat([runs, c1r], ignore_index=True),
            pd.concat([steps, c1s], ignore_index=True))


def check_sessions(label, runs, steps):
    """Answers and bytes must be identical to the earlier (shared-GPU) sessions."""
    for folder in CHECK_DIRS.get(label, []):
        f = ROOT / "outputs" / folder / "runs.csv"
        if not f.exists():
            continue
        old = pd.read_csv(f)
        m = runs.merge(old, on=["case_id", "variant"], suffixes=("", "_old"))
        if m.empty:
            continue
        same_text = (m["text"] == m["text_old"]).mean()
        max_db = (m["up_mb"] - m["up_mb_old"]).abs().max()
        print(f"[{label}] vs {folder}: {len(m)} runs, identical answers {same_text:.0%}, "
              f"max uplink difference {max_db:.4f} MB", flush=True)


def load_workload(label, folder, answer_tokens, arch="dream"):
    d = ROOT / "outputs" / folder
    runs, steps = pd.read_csv(d / "runs.csv"), pd.read_csv(d / "steps.csv")
    addon = ROOT / "outputs" / ADDON_DIRS.get(label, "__none__")
    if (addon / "runs.csv").exists():
        a_runs, a_steps = pd.read_csv(addon / "runs.csv"), pd.read_csv(addon / "steps.csv")
        new = sorted(set(a_runs["variant"]) - set(runs["variant"]))
        # sanity: same seed -> the add-on S0 should repeat the original S0 answers
        both = runs[runs.variant == "S0_split_full"].merge(
            a_runs[a_runs.variant == "S0_split_full"], on="case_id", suffixes=("", "_addon"))
        if len(both):
            same = (both["text"] == both["text_addon"]).mean()
            print(f"[{label}] add-on S0 repeats the original S0 text on {same:.0%} of "
                  f"{len(both)} cases", flush=True)
        runs = pd.concat([runs, a_runs[a_runs["variant"].isin(new)]], ignore_index=True)
        steps = pd.concat([steps, a_steps[a_steps["variant"].isin(new)]], ignore_index=True)
        extra = [c for c in ("edge_peak_gib", "cloud_peak_gib", "edge_energy_j", "cloud_energy_j")
                 if c in a_runs]
        if extra:
            runs = runs.drop(columns=[c for c in extra if c in runs]).merge(
                a_runs[["case_id", "variant"] + extra], on=["case_id", "variant"], how="left")
    for col in ("gen_refresh", "prompt_refresh"):
        if col in steps:
            steps[col] = to_bool(steps[col])
    runs, steps = derive_c1(label, runs, steps)
    check_sessions(label, runs, steps)
    rows = []
    for (case, var), st in steps.groupby(["case_id", "variant"], sort=False):
        r = runs[(runs.case_id == case) & (runs.variant == var)].iloc[0]
        cached = str(r.get("policy", "none")) != "none"
        e, c, w = run_flops(st, int(r.prompt_tokens), answer_tokens, cached, arch)
        rows.append({"workload": label, "case_id": case, "variant": var, "edge_tflops": e / 1e12,
                     "cloud_tflops": c / 1e12, "row_work": w})
    flops = pd.DataFrame(rows)
    runs = runs.assign(workload=label, ok=to_bool(runs["cache_ok"]))
    return runs, steps, flops


# ----------------------------------------------------------------- tables
def fmt(v, digits=1, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{100 * v:.{digits}f}\\%" if pct else f"{v:.{digits}f}"


def build_tables(all_runs, all_flops, emulated, answer_tokens):
    main, trade = [], []
    for label in WORKLOADS:
        runs = all_runs[all_runs.workload == label]
        fl = all_flops[all_flops.workload == label].groupby("variant").mean(numeric_only=True)
        em = emulated[label]
        ref = em[(em.bandwidth_mbps == REF_NET["bandwidth_mbps"]) & (em.rtt_ms == REF_NET["rtt_ms"])
                 & (em.edge_slowdown == REF_NET["edge_slowdown"])].set_index("variant")
        em10 = em[(em.bandwidth_mbps == 10) & (em.rtt_ms == REF_NET["rtt_ms"])
                  & (em.edge_slowdown == REF_NET["edge_slowdown"])].set_index("variant")
        acc = runs.groupby("variant")["ok"].mean()
        up = runs.groupby("variant")["up_mb"].mean()
        mem_e = runs.groupby("variant")["edge_peak_gib"].mean() if "edge_peak_gib" in runs else None
        en = (runs.groupby("variant")[["edge_energy_j", "cloud_energy_j"]].mean().sum(axis=1, min_count=2)
              if "edge_energy_j" in runs else None)
        order = ["cloud_only", "cloud_only_cache", "S0_split_full", "S1_split_exact_delta",
                 "S2_split_int8", "C2_cache_aware", "C4_cache_aware_int8", "A1_adaptive_cache_aware"]
        for v in order:
            src = {"cloud_only": "S0_split_full", "cloud_only_cache": "C1_cache_full"}.get(v, v)
            present = src in fl.index and v in ref.index
            lat = ref.loc[v, "mean_latency_s"] if present else np.nan
            edge_tf = np.nan if v.startswith("cloud_only") or not present else fl.loc[src, "edge_tflops"]
            cloud_tf = (fl.loc[src, "edge_tflops"] + fl.loc[src, "cloud_tflops"]
                        if v.startswith("cloud_only") and present
                        else (fl.loc[src, "cloud_tflops"] if present else np.nan))
            main.append({
                "workload": label, "method": NAMES[v], "variant": v,
                "accuracy": acc.get(src, np.nan),
                "edge_tflops": edge_tf, "cloud_tflops": cloud_tf,
                "uplink_mb": 0.0 if v.startswith("cloud_only") else up.get(src, np.nan),
                "cache_hit": (1 - fl.loc[src, "row_work"]) if present else np.nan,
                "latency_s": lat, "throughput_tok_s": answer_tokens[label] / lat if present else np.nan,
                "edge_peak_gib": (np.nan if v.startswith("cloud_only") or mem_e is None
                                  else mem_e.get(src, np.nan)),
                "energy_j": (np.nan if v.startswith("cloud_only") or en is None
                             else en.get(src, np.nan)),
            })
        if label == "GSM8K":
            for v, mode in [("S0_split_full", "S0 static split (bf16)"),
                            ("S1_split_exact_delta", "S1 exact delta"),
                            ("S2_split_int8", "S2 full boundary, INT8"),
                            ("C1_cache_full", "C1 cache, all rows"),
                            ("C2_cache_aware", "C2 cache-aware (bf16)"),
                            ("A1_adaptive_cache_aware", "A1 adaptive, cache-aware"),
                            ("C4_cache_aware_int8", "C4 cache-aware, INT8")]:
                present = v in ref.index
                trade.append({"mode": mode, "variant": v, "accuracy": acc.get(v, np.nan),
                              "uplink_mb": up.get(v, np.nan),
                              "latency_s_100mbps": ref.loc[v, "mean_latency_s"] if present else np.nan,
                              "latency_s_10mbps": em10.loc[v, "mean_latency_s"] if v in em10.index else np.nan})
    return pd.DataFrame(main), pd.DataFrame(trade)


def pending_note(main):
    missing = [] if main.workload.str.contains("LLaDA").any() else ["LLaDA rows"]
    if main["accuracy"][main.variant == "S2_split_int8"].isna().any():
        missing.append("the compressed static split")
    split_rows = main[~main.variant.str.startswith("cloud_only")]
    if split_rows["edge_peak_gib"].isna().any():
        missing.append("edge memory")
    if split_rows["energy_j"].isna().any():
        missing.append("energy")
    if not missing:
        return ""
    text = ", ".join(missing[:-1]) + (" and " if len(missing) > 1 else "") + missing[-1]
    return r"\hl{" + text[0].upper() + text[1:] + (" are" if len(missing) > 1 else " is") + " pending.}"


def main_table_tex(main):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Main results for Dream-v0-Instruct-7B"
             + (" and LLaDA-8B-Instruct" if main.workload.str.contains("LLaDA").any() else "")
             + r" (split after layer 4). Latency and throughput are emulated "
             r"at 100\,Mbit/s, 20\,ms RTT, edge $3\times$ slower than an A6000. FLOPs are analytic (per answer). "
             r"Cache hit is $1-w$ from Eq.~\eqref{eq:rowwork}. Quality is accuracy on 30 GSM8K problems and on 12 authored "
             r"long-prompt cases (instruction cases checked for format). Cloud-only rows run all layers on one GPU. "
             r"Edge GiB is the peak allocated memory on the edge GPU (weights included); energy is the NVML "
             r"energy of both GPUs per answer, excluding the emulated link. " + pending_note(main) + "}",
             r"\label{tab:main-results}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llccccccccc}", r"\toprule",
             r"Workload & Method & Quality $\uparrow$ & Edge TFLOPs $\downarrow$ & Cloud TFLOPs $\downarrow$ & "
             r"Uplink MB $\downarrow$ & Cache hit $\uparrow$ & Latency s $\downarrow$ & Throughput tok/s $\uparrow$ & "
             r"Edge GiB $\downarrow$ & Energy J $\downarrow$ \\", r"\midrule"]
    for i, (label, grp) in enumerate(main.groupby("workload", sort=False)):
        if i:
            lines.append(r"\midrule")
        for _, r in grp.iterrows():
            pending = np.isnan(r.latency_s)
            cells = [DISPLAY.get(label, label) if _ == grp.index[0] else "", r.method,
                     fmt(r.accuracy, 1, pct=True), fmt(r.edge_tflops, 0), fmt(r.cloud_tflops, 0),
                     "$<$0.01" if r.variant.startswith("cloud_only") else fmt(r.uplink_mb, 1),
                     fmt(r.cache_hit, 0, pct=True), fmt(r.latency_s, 1), fmt(r.throughput_tok_s, 2),
                     fmt(r.edge_peak_gib, 1), fmt(r.energy_j, 0)]
            if pending:
                cells = cells[:2] + [r"\hl{pending}"] + ["--"] * (len(cells) - 3)
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def trade_table_tex(trade):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Quality--systems trade-off on GSM8K ($n=30$, Dream, split after layer 4). "
             r"Latency emulated with 20\,ms RTT and an edge $3\times$ slower than an A6000. "
             r"Configurations as in Table~\ref{tab:configs}. C1 performs C2's computation and sends every row; "
             r"its latency uses C2's measured computation.}",
             r"\label{tab:tradeoff}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Latency (s) $\downarrow$} \\ \cmidrule(lr){4-5}",
             r"Config. & Acc. $\uparrow$ & Uplink MB $\downarrow$ & 100\,Mbit/s & 10\,Mbit/s \\", r"\midrule"]
    for _, r in trade.iterrows():
        if np.isnan(r.latency_s_100mbps):
            lines.append(f"{r['mode']} & \\hl{{pending}} & -- & -- & -- \\\\")
        else:
            lines.append(f"{r['mode']} & {fmt(r.accuracy, 1, pct=True)} & {fmt(r.uplink_mb, 1)} & "
                         f"{fmt(r.latency_s_100mbps, 1)} & {fmt(r.latency_s_10mbps, 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------- figures
# Categorical palette (validated, light mode): blue, orange, aqua; baselines in neutral gray.
STYLE = {
    "S0_split_full": dict(color="#6f6e69", marker="s", ls="-", label="Static split"),
    "C2_cache_aware": dict(color="#2a78d6", marker="o", ls="-", label="DASR exact", open=True),
    "A1_adaptive_cache_aware": dict(color="#1baf7a", marker="^", ls="-", label="DASR adaptive"),
    "C4_cache_aware_int8": dict(color="#eb6834", marker="D", ls="-", label="DASR INT8"),
}
REFS = {"cloud_only": ("Cloud-only", (0, (4, 3))), "cloud_only_cache": ("Cloud-only + cache", (0, (1, 2)))}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def fig_latency_bandwidth(plt, emulated, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    for ax, (label, em) in zip(axes, emulated.items()):
        sub = em[(em.rtt_ms == REF_NET["rtt_ms"]) & (em.edge_slowdown == REF_NET["edge_slowdown"])]
        for v, st in STYLE.items():
            s = sub[sub.variant == v].sort_values("bandwidth_mbps")
            if s.empty:
                continue
            ax.plot(s.bandwidth_mbps, s.mean_latency_s, color=st["color"], marker=st["marker"],
                    ms=6 if st.get("open") else 4, lw=1.6, label=st["label"],
                    mfc="none" if st.get("open") else st["color"], mew=1.2,
                    zorder=4 if st.get("open") else 3)
            ax.fill_between(s.bandwidth_mbps, s.ci95_low, s.ci95_high, color=st["color"], alpha=0.12, lw=0)
        for v, (name, dash) in REFS.items():
            val = sub[sub.variant == v].mean_latency_s.mean()
            ax.axhline(val, color="#3f3e3a", lw=0.9, ls=dash, label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Link throughput (Mbit/s)")
        ax.set_title(f"{label}: RTT {REF_NET['rtt_ms']} ms, edge {REF_NET['edge_slowdown']:g}x slower",
                     loc="left")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Latency per answer (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(plt, main, path):
    g = main[(main.workload == "GSM8K") & main.variant.isin(list(STYLE))].dropna(subset=["latency_s"])
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for ax, (col, xlabel) in zip(axes, [("latency_s", "Latency (s), 100 Mbit/s"),
                                        ("uplink_mb", "Uplink per answer (MB)")]):
        for _, r in g.iterrows():
            st = STYLE[r.variant]
            if st.get("open"):
                ax.scatter(r[col], 100 * r.accuracy, facecolor="none", edgecolor=st["color"],
                           marker=st["marker"], s=70, linewidth=1.3, zorder=4, label=st["label"])
            else:
                ax.scatter(r[col], 100 * r.accuracy, color=st["color"], marker=st["marker"], s=34,
                           zorder=3, edgecolor="#fcfcfb", linewidth=0.8, label=st["label"])
        ax.margins(x=0.15, y=0.25)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e6e5e0", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("GSM8K accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_uplink_timeline(plt, steps_by_label, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for ax, (label, steps) in zip(axes, steps_by_label.items()):
        case = steps.case_id.iloc[0]
        # INT8 first (bottom), exact on top: refresh steps send every row in bf16 for both
        # DASR variants, so their spikes coincide; static split is dashed on top.
        order = (("C4_cache_aware_int8", 1.0, "-", 2), ("C2_cache_aware", 1.0, "-", 3),
                 ("S0_split_full", 1.3, (0, (4, 2)), 4))
        for v, lw, ls, z in order:
            st = STYLE[v]
            s = steps[(steps.case_id == case) & (steps.variant == v)].sort_values("step")
            if s.empty:
                continue
            ax.plot(s.step, s.up_bytes / 1e6, color=st["color"], lw=lw, ls=ls, label=st["label"],
                    drawstyle="steps-mid", zorder=z)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 12)
        ax.set_yticks([0.5, 1, 2, 5, 10])
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Denoising step")
        ax.set_title(f"{label} ({case})", loc="left")
        ax.grid(True, color="#e6e5e0", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Uplink per step (MB)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs, all_flops, emulated, steps_by_label, answer_tokens = [], [], {}, {}, {}
    for label, (folder, tokens, arch) in WORKLOADS.items():
        runs, steps, flops = load_workload(label, folder, tokens, arch)
        _, summary = emu.emulate(steps, runs)
        # full emulation grid (incl. C1 derived from C2): source of the latency table
        summary.to_csv(OUT / f"emulated_summary_{folder}.csv", index=False)
        all_runs.append(runs); all_flops.append(flops)
        emulated[label], steps_by_label[label], answer_tokens[label] = summary, steps, tokens
    all_runs, all_flops = pd.concat(all_runs, ignore_index=True), pd.concat(all_flops, ignore_index=True)
    all_flops.to_csv(OUT / "flops_per_run.csv", index=False)
    main_t, trade_t = build_tables(all_runs, all_flops, emulated, answer_tokens)
    main_t.to_csv(OUT / "main_table.csv", index=False)
    trade_t.to_csv(OUT / "tradeoff_table.csv", index=False)
    (OUT / "main_table.tex").write_text(main_table_tex(main_t), encoding="utf-8")
    (OUT / "tradeoff_table.tex").write_text(trade_table_tex(trade_t), encoding="utf-8")
    plt = setup_mpl()
    fig_latency_bandwidth(plt, {k: emulated[k] for k in DREAM_LABELS},
                          OUT / "fig_latency_bandwidth.pdf")
    fig_tradeoff(plt, main_t, OUT / "fig_tradeoff_gsm8k.pdf")
    fig_uplink_timeline(plt, {k: steps_by_label[k] for k in DREAM_LABELS},
                        OUT / "fig_uplink_timeline.pdf")
    pd.set_option("display.width", 200)
    print(main_t.round(3).to_string(index=False))
    print()
    print(trade_t.round(3).to_string(index=False))
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
