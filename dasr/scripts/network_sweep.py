"""Post-process measured compute under an analytical network model, never RTT measurements."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=["id", "bandwidth_mbps", "rtt_ms", "modeled_latency_ms", "application_bytes", "kind"])
        writer.writeheader()
        for row in rows:
            cfg, totals = row["config"], row["totals"]
            if cfg["sleep_network"]:
                raise ValueError("Use logs without injected network sleep to avoid double-counting")
            messages = 1 if totals.get("request_once_cloud") else sum(int(s["uplink_application_bytes"] > 0) for s in row["steps"])
            for bandwidth in (1, 5, 10, 25, 50, 100, 1000):
                for rtt in (5, 20, 50, 100):
                    network = rtt * messages + 8 * totals["application_bytes"] / (bandwidth * 1000)
                    writer.writerow(dict(id=row["id"], bandwidth_mbps=bandwidth, rtt_ms=rtt,
                                         modeled_latency_ms=totals["latency_ms"] + network,
                                         application_bytes=totals["application_bytes"], kind="analytical_network_emulation"))


if __name__ == "__main__":
    main()
