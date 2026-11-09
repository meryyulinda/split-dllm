"""Exp10A: quality robustness of official dLLM-Cache on Dream.

Run from dasr1/: python experiments/exp10a_official_feature_cache_quality.py
Requires previous experiment file:
    experiments/exp9b_official_feature_cache_diagnostics.py

Controlled pilot (NOT a public benchmark): 4 original, coherent case documents,
3 tasks per document (factual, multi-step arithmetic, constrained instruction),
3 seeds => 36 paired cases. Reports *raw outputs* and offline review fields.
No LLM-as-judge; the arithmetic heuristic is advisory, not ground truth.
Optional B-C-B timing is separate from token-trajectory instrumentation.
"""
from __future__ import annotations

import json
import os
import platform
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import transformers

import configs.experiment_config as cfg
from src.model_loader import load_dream_model
from exp9b_official_feature_cache_diagnostics import (
    import_official, prepare_input, run_generation, compare_history,
)


def setting(key, fallback):
    return getattr(cfg, key, fallback)


SEEDS = list(setting("EXP10A_SEEDS", [cfg.SEED, cfg.SEED + 1, cfg.SEED + 2]))
N_TOKENS = int(setting("EXP10A_NEW_TOKENS", 128))
STEPS = int(setting("EXP10A_STEPS", 128))
CACHE_CFG = tuple(setting("EXP10A_CACHE_CONFIG", (100, 7, 0.25)))
WARMUP_RUNS = int(setting("EXP10A_WARMUP_RUNS", 1))
RUN_TIMING = bool(setting("EXP10A_RUN_TIMING", False))
TIMING_REPEATS = int(setting("EXP10A_TIMING_REPEATS", 2))
# Optional first-pass pilot, e.g. ["library_factual", "library_math", "library_instruction"].
CASE_IDS = list(setting("EXP10A_CASE_IDS", []))
OUTPUT_DIR = ROOT / "outputs" / "exp10a"


# Authored self-contained contexts, not genuine industrial documents / LongBench.
# Facts and arithmetic were deliberately chosen to be independently checkable.
DOCUMENTS = {
    "library": """
RIVERTON LIBRARY DIGITAL ACCESS PROGRAM — OPERATIONS NOTE

The Riverton library network is preparing a public digital catalogue to let
residents search older local-history materials before travelling to a branch.
The project is not attempting to place every item online. Its initial scope
is to create reliable catalogue entries for photographs, newsletters, oral
history recordings, and municipal planning folders. Items covered by
copyright restrictions may have searchable descriptions without public
scans. Staff have agreed that the catalogue should state clearly whenever
a physical visit or permission is required to view the underlying document.

The rollout coordinator is Mara Chen. She chairs a weekly meeting with the
cataloguing librarians, accessibility volunteers, the small IT team, and the
records officer. The public launch is planned for 14 September 2026, subject
to a final accessibility review. The team will not treat a successful import
as sufficient evidence that public search works. Before launch, volunteers
will inspect navigation by keyboard, screen-reader reading order, descriptive
text for images, and the clarity of the request-a-copy form. The records
officer has authority to remove an item from the public index if its metadata
reveals protected personal information.

The first intake report describes 4,200 North Branch records, 3,100 South
Branch records, and 2,700 East Branch records. A record is the catalogue
unit, regardless of whether its associated item contains multiple pages.
Across the three branches, 8,200 records are already fully reviewed and
480 are flagged for a specialist privacy or rights decision. The remaining
1,320 records form the ordinary scanning backlog for a separate staffing
exercise. Flagged records are not included in this ordinary backlog and
should not be counted as automatically cleared when scanning work increases.

For the staffing estimate, four technicians are assigned exclusively to the
1,320-record ordinary backlog. Each technician is expected to complete 35
records per working day, and management assumes six working days without
absences or machine downtime. A completed scan leaves the ordinary backlog
immediately; the team does not apply a second acceptance percentage in this
specific arithmetic exercise. Management wants the remaining-backlog number,
not the total records in the library network, and does not want the flagged
privacy-review queue mixed into the calculation.

The intake process begins with a condition check and a temporary accession
identifier. Physical items that are fragile are routed to conservation rather
than put directly through a scanner. After scanning, cataloguers verify the
title, date range, and rights note against the physical folder. Accessibility
volunteers review images and the catalogue interface, but they are not
allowed to publish personal information found in source materials. Public
volunteers are instructed to report uncertain content instead of silently
changing dates or inventing missing details.

A pilot search workshop identified several practical issues. People sometimes
assumed an online title meant the digitized item was immediately available.
Some scanned photographs were difficult to understand without captions, and
some folder titles contained names that should have remained private. The
team therefore added a visible access-status field, a mandatory description
review, and a privacy escalation form. Each proposed change has to be logged
with the item identifier and the reviewer so another staff member can trace
who approved it. If the workshop raises new accessibility failures, the
launch coordinator may delay the public release rather than waive the check.

The volunteer briefing should distinguish intake, accessibility, and privacy
work. Intake covers identifiers and item condition; accessibility covers
navigation and meaningful alternative descriptions; privacy covers possible
personal data and routing uncertain cases to the records officer. Nobody may
promise that all source scans will be downloadable on launch day. The network
plans to publish a follow-up report after its first month of public use.
""",
    "sensors": """
SEABROOK COASTAL FLOOD SENSOR PILOT — OPERATIONS BRIEF

Seabrook is running a pilot of low-cost water-level sensors along drainage
channels and tidal access roads. The goal is early warning and maintenance
planning, not a claim that every local flood can be predicted. Engineers
compare sensor signals with staff observations, tide records, and rainfall
measurements before advising the emergency coordination desk. A reading from
one sensor is never, by itself, grounds for ordering an evacuation. During
storm warnings the desk receives time-stamped updates with a location code
and an indicator for whether a technician verified the device recently.

The deployed network has 18 active sensors in the western sector, 12 in the
eastern sector, and 10 in the central sector. An additional eight sensors are
kept in sealed boxes as replacement stock and are not counted among active
transmitting devices. The sensors are configured to send six readings per
hour while active. For a controlled communication-load exercise, assume every
active sensor reports continuously for five hours without an outage and
without retransmission. The exercise asks for the total original readings
from active sensors during that five-hour interval, not packet bytes, reserve
devices, or duplicate network retries.

A field crew carries out a scheduled inspection every 14 days. Inspections
include battery condition, enclosure seals, attachment to the mounting pole,
clock synchronization, and a comparison with a manual staff gauge. Engineers
mark readings as suspect when the enclosure is submerged in debris or a
sensor reports the same measurement for an implausibly long period. The
maintenance log retains both original and corrected observations, and
any deletion requires review from the operations lead. The operations lead
is Tomas Reyes, who also manages the stock of replacement devices.

When the municipal weather desk issues a severe-storm notice, the monitoring
team enters a higher-attention protocol. Operators first verify suspicious
readings against nearby devices and, where safe, a field observation.
If the cellular link becomes unreliable, the emergency desk switches to its
backup radio channel and marks the affected locations as having reduced
automated visibility. Crew members must not enter closed roads or flooded
areas merely to recover a sensor. A field report includes the time, site
identifier, observed water level when safe to collect, and the reporter's
name. The desk escalates confirmed rapid rises according to the city's
existing incident plan.

The data pipeline receives sensor messages through a gateway and assigns
sequence numbers to identify duplicate transmissions. A downstream dashboard
displays a confidence marker when packet arrivals are delayed. Historical
plots retain gaps rather than drawing a continuous line across missing
observations. Analysts can compare sectors but should not treat unequal
sampling coverage as equal flood exposure. Training sessions emphasize that
sensor telemetry supports human judgement and that a normal dashboard does
not guarantee road safety.

Residents can subscribe to municipal alerts through the public safety site;
the pilot does not collect personal telephone numbers inside sensor packets.
A final report will evaluate maintenance burden, false alarms, communication
reliability, and how quickly operators noticed actual threshold crossings.
The decision about a wider deployment will consider these operational costs,
not merely the total volume of collected readings.
""",
    "clinic": """
NORTHLIGHT CLINIC VACCINE INVENTORY — STAFF PROCEDURE

Northlight Clinic maintains a small temperature-controlled inventory for
routine appointments and for outreach days at nearby community centres.
The inventory manager, Imani Patel, checks deliveries against supplier
manifests and records product, lot, receipt time, expiry date, and refrigerator
location. The clinic uses a digital stock register, but physical counts are
still required when a shipment arrives or a storage alarm occurs. A
manufacturer's packing list is not a substitute for an actual count.

For a controlled scheduling exercise, the clinic receives 15 sealed boxes,
with 24 usable doses per box. All doses are identical for the purpose of this
arithmetic problem and no box is partly filled. Before patient scheduling,
40 doses from the received shipment are reserved for a staff training session.
The problem assumes no wastage, breakage, returns, or other reservations.
The task is to calculate doses remaining for patient appointments after this
training allocation; it does not ask for current stock from earlier shipments.
Do not subtract expired stock from this exercise because the delivered boxes
are defined as in date and usable.

The clinic's reorder threshold for appointment-available doses is 140. Staff
should check whether future appointments can be covered and record a purchase
request when available doses fall below that threshold. This is an operational
trigger, not a regulatory minimum for all clinics. The inventory manager
reviews threshold exceptions when a delivery is delayed. Every adjustment
in the register requires a reason and a named reviewer so that a later
reconciliation can reconstruct the sequence of changes.

Temperature control is handled separately from the appointment-count exercise.
Receiving staff verify that the delivery monitor remained within the
manufacturer's specified range and note any excursion immediately. A shipment
with an unresolved excursion is placed on hold until the responsible
pharmacist assesses it. The clinic will not restore stock to available status
solely because the refrigerator now displays a normal temperature. If
products have multiple lots, staff preserve lot-level traceability instead
of merging the balances into a single unidentified pool.

When preparing an appointment tray, a nurse confirms the required product,
its lot number, and the expiration date. The oldest eligible stock is
normally used first, but the responsible pharmacist may prioritize a
particular lot following a safety notice. Returned doses are not assumed
to be reusable until storage conditions have been verified. For community
outreach, transport containers must be logged before departure and again on
return. The receiving location should know whom to contact if a temperature
logger fails in transit.

Training emphasizes three checks before accepting new supplies into the
available inventory: verify temperature evidence, preserve lot traceability,
and confirm the expiry status. Staff should document discrepancies instead
of adjusting the count to match the manifest automatically. This procedure
supports patient safety and auditability; it does not replace professional
clinical judgement or the manufacturer's product-specific handling rules.
""",
    "transit": """
GREENRIDGE CITY BUS NETWORK — SERVICE OPERATIONS MEMO

Greenridge's transit office is reviewing how it communicates service changes
to riders during a six-week roadwork programme. The office runs buses on a
small loop network serving a train station, two schools, a clinic, the civic
centre, and several residential areas. The memo concerns the scheduled
weekday network, not special-event shuttles. A service desk receives calls
about missed connections, wheelchair access, passenger information, and
which stops have temporary boarding areas.

For a controlled mileage estimate, the operator has seven buses assigned to
loop service. Every assigned bus is scheduled to complete five full loops
in one day. Each full loop is 18 kilometres long. Assume that all seven buses
complete all five loops, with no additional depot mileage, deadheading, or
route diversions. The task asks for total daily kilometres operated by the
seven buses combined, rather than the distance of one bus or one loop. The
city uses this simplified number for a first-pass maintenance-planning
exercise; real trip records would include delays and non-revenue travel.

The rider feedback deadline for the current roadwork consultation is
23 November 2026. The transit office will accept comments through the city
website, at the civic centre counter, and by telephone during service-desk
hours. The review manager, Elena Torres, will summarize comments about
boarding safety and accessibility before recommending any permanent stop
changes. The date is a deadline for comments, not the start or end date of
the roadwork itself. Operators should not suggest that a temporary route
has become permanent until the office publishes a formal decision.

Construction may occasionally delay buses near Market Street. Two stops may
move to accessible temporary boarding islands if crews need the curb space.
The transit office requires that temporary locations include detectable
boarding edges, legible signs, and a safe path from the pavement to the
platform. When these features are unavailable, the office must describe the
accessibility limitation instead of presenting the substitute stop as fully
accessible. Riders with mobility concerns can contact the service desk for
the latest boarding arrangements.

The control room tracks service interruptions and publishes status notices
that separate verified changes from preliminary contractor reports. A note
about a possible delay is not equivalent to a confirmed cancellation.
Dispatchers record the vehicle, approximate time, location, and source of
the report. For disruptions likely to affect hospital connections, the office
also notifies the local accessibility advisory group. The public notice
should avoid blaming drivers for delays caused by road closures.

Rider-facing messages should be plain and short. They should mention the
possibility of roadwork-related delays, accessibility or temporary boarding
information, and a contact option for riders to verify current conditions.
The notice must not promise exact arrival times or claim all temporary stops
are barrier-free before field verification. Feedback is important, but the
memo's central purpose is to give riders actionable information about their
next trip and how to ask for assistance.
""",
}


# One document is reused for three controlled task types. This is a pilot,
# not 12 independent long-context sources or a representative quality benchmark.
CASE_SPECS = [
    ("library_factual", "library", "factual",
     "Who is the rollout coordinator? Answer with the name only.", "Mara Chen"),
    ("library_math", "library", "math",
     "Starting from the ordinary scanning backlog, how many records remain after all four technicians work for six days? Give the final number of records.", 480),
    ("library_instruction", "library", "instruction",
     "Write exactly three numbered action items for volunteers, covering intake, accessibility, and privacy in that order. Use only this note.", None),
    ("sensors_factual", "sensors", "factual",
     "How often is a scheduled field inspection performed? Answer with the interval only.", "14 days"),
    ("sensors_math", "sensors", "math",
     "How many original readings will all active sensors transmit over five hours? Give the final number of readings.", 1200),
    ("sensors_instruction", "sensors", "instruction",
     "Write exactly three numbered actions for severe-storm monitoring: reading verification, backup communication, and incident reporting, in that order.", None),
    ("clinic_factual", "clinic", "factual",
     "What is the reorder threshold for appointment-available doses? Answer with the quantity only.", "140"),
    ("clinic_math", "clinic", "math",
     "How many doses from the received shipment remain for patient appointments after the training reservation? Give the final number of doses.", 320),
    ("clinic_instruction", "clinic", "instruction",
     "Write exactly three numbered receiving checks, covering temperature evidence, lot tracking, and expiry status in that order.", None),
    ("transit_factual", "transit", "factual",
     "What is the rider feedback deadline? Answer with the date only.", "23 November 2026"),
    ("transit_math", "transit", "math",
     "How many kilometres do all seven buses operate in total on the specified day? Give the final number of kilometres.", 630),
    ("transit_instruction", "transit", "instruction",
     "Write exactly three numbered rider-advisory points, covering potential delays, accessible boarding information, and how to contact the service desk, in that order.", None),
]


def make_cases():
    cases = []
    for case_id, doc_key, task, question, gold in CASE_SPECS:
        prompt = (
            "Read the following operations note. Treat it as the only source of facts. "
            "Do not use information from other cases.\n\n"
            + DOCUMENTS[doc_key].strip()
            + "\n\nQUESTION:\n" + question
        )
        cases.append({"case_id": case_id, "doc_id": doc_key, "task": task,
                      "prompt": prompt, "question": question,
                      "gold_answer": gold})
    return cases


def final_numeric_answer(answer):
    """Advisory extraction: last written number; not proof of correctness.

    Example: '15 * 12 is 180. The answer is 181.' => 181 (incorrect).
    """
    nums = re.findall(r"(?<![\w])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)", answer)
    return float(nums[-1].replace(",", "")) if nums else None


def advisory_score(case, answer):
    if case["task"] == "math":
        observed = final_numeric_answer(answer)
        return {"advisory_gold_match": observed == float(case["gold_answer"]),
                "advisory_extracted_answer": observed, "instruction_format_pass": None}
    if case["task"] == "factual":
        norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
        return {"advisory_gold_match": norm(case["gold_answer"]) in norm(answer),
                "advisory_extracted_answer": None, "instruction_format_pass": None}
    # Format-only heuristic, not semantic correctness.
    numbered = re.findall(r"(?m)^\s*[123][.)]\s+", answer.strip())
    return {"advisory_gold_match": None, "advisory_extracted_answer": None,
            "instruction_format_pass": len(numbered) == 3}


def save_rows(rows, name):
    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / name, index=False)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Exp10A requires an NVIDIA CUDA GPU")
    if "Dream" not in cfg.MODEL_NAME:
        raise RuntimeError("Exp10A uses the official Dream cache adapter only")
    if N_TOKENS < 1 or STEPS < 1 or TIMING_REPEATS < 1:
        raise ValueError("Invalid generation/timing configuration")
    if len(CACHE_CFG) != 3:
        raise ValueError("EXP10A_CACHE_CONFIG must be (prompt_interval, gen_interval, ratio)")
    if int(CACHE_CFG[0]) < 1 or int(CACHE_CFG[1]) < 1 or not 0 < float(CACHE_CFG[2]) <= 1:
        raise ValueError("Invalid official cache settings")
    cases = make_cases()
    if CASE_IDS:
        unknown = set(CASE_IDS) - {c["case_id"] for c in cases}
        if unknown:
            raise KeyError(f"Unknown case IDs: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in CASE_IDS]
    if not cases or not SEEDS:
        raise ValueError("No cases or seeds selected")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    repo, api = import_official()
    model, tokenizer = load_dream_model(cfg.MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    print("EXP10A — official dLLM-Cache quality robustness | Dream | single GPU")
    print("cases", len(cases), "seeds", SEEDS, "generation", N_TOKENS, STEPS,
          "cache", CACHE_CFG, "timing", RUN_TIMING, flush=True)

    fid_rows, answer_rows, timing_rows = [], [], []
    total = len(cases) * len(SEEDS)
    if WARMUP_RUNS:
        ids_w, mask_w = prepare_input(tokenizer, cases[0]["prompt"], device)
        for _ in range(WARMUP_RUNS):
            run_generation(model, ids_w, mask_w, N_TOKENS, STEPS, SEEDS[0], device)
            run_generation(model, ids_w, mask_w, N_TOKENS, STEPS, SEEDS[0], device,
                           cache_api=api, cache_cfg=CACHE_CFG)
        print("warmup done", flush=True)

    for case_index, case in enumerate(cases):
        ids, mask = prepare_input(tokenizer, case["prompt"], device)
        plen = int(ids.shape[1])
        print(f"\n{case['case_id']} | prompt tokens={plen}", flush=True)
        for seed_index, seed in enumerate(SEEDS):
            index = case_index * len(SEEDS) + seed_index + 1
            # Fidelity only. Each call resets its RNG; cache is fresh per call.
            base = run_generation(model, ids, mask, N_TOKENS, STEPS, seed, device,
                                  capture_trajectory=True)
            cached = run_generation(model, ids, mask, N_TOKENS, STEPS, seed, device,
                                    cache_api=api, cache_cfg=CACHE_CFG,
                                    capture_trajectory=True)
            control = run_generation(model, ids, mask, N_TOKENS, STEPS, seed, device,
                                     capture_trajectory=True)
            comparison = compare_history(base, cached, plen)
            baseline_control = compare_history(base, control, plen)
            baseline_text = tokenizer.decode(base["generated_ids"], skip_special_tokens=True)
            cached_text = tokenizer.decode(cached["generated_ids"], skip_special_tokens=True)
            bscore = advisory_score(case, baseline_text)
            cscore = advisory_score(case, cached_text)
            record = {"case_id": case["case_id"], "doc_id": case["doc_id"],
                      "task": case["task"], "seed": int(seed),
                      "prompt_tokens": plen, "new_tokens": N_TOKENS, "steps": STEPS,
                      "baseline_control_exact": baseline_control["exact_trajectory"],
                      **comparison}
            fid_rows.append(record)
            answer_rows.append({
                "case_id": case["case_id"], "task": case["task"], "seed": int(seed),
                "question": case["question"], "gold_answer": case["gold_answer"],
                "baseline_text": baseline_text, "cache_text": cached_text,
                "exact_final_text": baseline_text == cached_text,
                "character_similarity_not_quality": SequenceMatcher(
                    None, baseline_text, cached_text).ratio(),
                "baseline_advisory_gold_match": bscore["advisory_gold_match"],
                "cache_advisory_gold_match": cscore["advisory_gold_match"],
                "baseline_extracted_numeric": bscore["advisory_extracted_answer"],
                "cache_extracted_numeric": cscore["advisory_extracted_answer"],
                "baseline_instruction_format_pass": bscore["instruction_format_pass"],
                "cache_instruction_format_pass": cscore["instruction_format_pass"],
                "baseline_correct_human": "", "cache_correct_human": "",
                "quality_regression_human": "", "review_notes": "",
                "baseline_control_exact": baseline_control["exact_trajectory"],
            })
            save_rows(fid_rows, "fidelity_runs.csv")
            save_rows(answer_rows, "answer_quality_review.csv")
            tag = "EXACT" if comparison["exact_trajectory"] else (
                f"DIFF@{comparison['first_divergent_step']}")
            print(f"[{index:02d}/{total:02d}] {case['case_id']:21s} "
                  f"seed={seed} {tag} final={comparison['exact_final_ids']} "
                  f"baseline_control={baseline_control['exact_trajectory']} "
                  f"gold_advisory={cscore['advisory_gold_match']}", flush=True)

            if RUN_TIMING:
                for rep in range(TIMING_REPEATS):
                    b1 = run_generation(model, ids, mask, N_TOKENS, STEPS, seed, device)
                    c = run_generation(model, ids, mask, N_TOKENS, STEPS, seed, device,
                                       cache_api=api, cache_cfg=CACHE_CFG)
                    b2 = run_generation(model, ids, mask, N_TOKENS, STEPS, seed, device)
                    b_ms = (b1["ms"] + b2["ms"]) / 2
                    timing_rows.append({
                        "case_id": case["case_id"], "task": case["task"],
                        "seed": int(seed), "repeat": rep, "prompt_tokens": plen,
                        "baseline_pair_stable": torch.equal(
                            b1["generated_ids"], b2["generated_ids"]),
                        "baseline_timed_vs_fidelity_equal": torch.equal(
                            b1["generated_ids"], base["generated_ids"]),
                        "cache_timed_vs_fidelity_equal": torch.equal(
                            c["generated_ids"], cached["generated_ids"]),
                        "baseline_before_ms": b1["ms"],
                        "baseline_after_ms": b2["ms"],
                        "paired_baseline_ms": b_ms, "cache_ms": c["ms"],
                        "cache_setup_ms": c["cache_setup_ms"],
                        "saving_pct": 100 * (1 - c["ms"] / b_ms),
                        "including_setup_saving_pct": 100 * (
                            1 - (c["ms"] + c["cache_setup_ms"]) / b_ms),
                    })
                    save_rows(timing_rows, "timing_runs.csv")

    df = pd.DataFrame(fid_rows)
    df.groupby("task", sort=True).agg(
        cases=("case_id", "size"),
        exact_trajectory=("exact_trajectory", "sum"),
        exact_final_ids=("exact_final_ids", "sum"),
        stable_baseline_controls=("baseline_control_exact", "sum"),
    ).reset_index().to_csv(OUTPUT_DIR / "fidelity_by_task.csv", index=False)
    if timing_rows:
        td = pd.DataFrame(timing_rows)
        td = td[td["baseline_pair_stable"]]
        if not td.empty:
            td.groupby("task", sort=True).agg(
                trials=("saving_pct", "size"),
                median_saving_pct=("saving_pct", "median"),
                median_including_setup_saving_pct=("including_setup_saving_pct", "median"),
            ).reset_index().to_csv(OUTPUT_DIR / "timing_by_task.csv", index=False)
    metadata = {
        "experiment": "Exp10A controlled authored-context quality pilot",
        "model": cfg.MODEL_NAME, "seeds": SEEDS,
        "new_tokens": N_TOKENS, "steps": STEPS,
        "cache_config": CACHE_CFG, "official_repo": str(repo),
        "cases": [c["case_id"] for c in cases],
        "num_independent_documents": len({c["doc_id"] for c in cases}),
        "timing": RUN_TIMING, "timing_repeats": TIMING_REPEATS,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name(device),
        "limitations": "Coherent authored notes, not real documents/LongBench. 12 tasks share 4 contexts. Manual quality review required. Timings exclude cache initialization but setup time is recorded; no two-GPU or edge-cloud transfer.",
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("\nExp10A completed. Output:", OUTPUT_DIR, flush=True)
    print("Review answer_quality_review.csv; do not treat advisory scores as final correctness.")
    print(df.groupby("task")["exact_final_ids"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
