"""GPU energy (NVML) helpers shared by the two-GPU experiments (copied from Exp11)."""
import os
import time

import torch


class EnergyMeter:
    """NVML total-energy counters for the edge and cloud GPUs (None if unavailable)."""

    def __init__(self, devices):
        self.handles = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
        except Exception as exc:  # pragma: no cover - depends on the machine
            print(f"[energy] pynvml unavailable ({exc}); energy columns stay empty. "
                  f"pip install nvidia-ml-py to enable.", flush=True)
            return
        visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                   if x.strip()]
        for name, dev in devices.items():
            handle = None
            try:  # robust: match by UUID
                uuid = str(torch.cuda.get_device_properties(dev).uuid)
                handle = pynvml.nvmlDeviceGetHandleByUUID(
                    uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
            except Exception:
                try:  # fallback: CUDA_VISIBLE_DEVICES=1,2 -> logical 0 = physical 1
                    idx = dev.index or 0
                    phys = int(visible[idx]) if visible else idx
                    handle = pynvml.nvmlDeviceGetHandleByIndex(phys)
                except Exception as exc:
                    print(f"[energy] no NVML handle for {name} ({dev}): {exc}", flush=True)
            if handle is not None:
                try:
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    self.handles[name] = handle
                except Exception as exc:
                    print(f"[energy] {name}: energy counter not supported ({exc})", flush=True)

    def read_j(self):
        return {n: self.nvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1e3
                for n, h in self.handles.items()}


def measure_idle_w(meter, devices, seconds):
    if not meter.handles or seconds <= 0:
        return {}
    for d in devices:
        torch.cuda.synchronize(d)
    e0, t0 = meter.read_j(), time.perf_counter()
    time.sleep(seconds)
    e1, dt = meter.read_j(), time.perf_counter() - t0
    return {n: (e1[n] - e0[n]) / dt for n in e0}
