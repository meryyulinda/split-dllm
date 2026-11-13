import pandas as pd
import matplotlib.pyplot as plt

import os as _os_sfx
import sys as _sys_sfx
_sys_sfx.path.append(_os_sfx.path.dirname(_os_sfx.path.dirname(_os_sfx.path.abspath(__file__))))
import configs.experiment_config as _sfx_cfg  # noqa: E402
SFX = getattr(_sfx_cfg, "OUTPUT_SUFFIX", "")  # "" = original folders, e.g. "_entropy"


df = pd.read_csv(f"outputs/exp1{SFX}/token_changes.csv")

plt.figure()

plt.plot(df["transition"], df["changed_ratio"],)

plt.xlabel("Diffusion transition")

plt.ylabel("Changed-token ratio")

plt.title("Token Changes Across Dream Denoising transitions")

plt.tight_layout()

plt.savefig(f"outputs/exp1{SFX}/changed_ratio.png", dpi=200)

plt.show()