import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/exp1/token_changes.csv")

plt.figure()

plt.plot(df["transition"], df["changed_ratio"],)

plt.xlabel("Diffusion transition")

plt.ylabel("Changed-token ratio")

plt.title("Token Changes Across Dream Denoising transitions")

plt.tight_layout()

plt.savefig("outputs/exp1/changed_ratio.png", dpi=200)

plt.show()