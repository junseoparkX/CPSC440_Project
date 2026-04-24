from pathlib import Path
import numpy as np
import pandas as pd

# base paths
ROOT = Path(__file__).resolve().parent.parent   # repo root
DATA = ROOT / "data"

# load npy files (all 3)
pop_npy     = np.load(DATA / "ga_pop.npy",     allow_pickle=True)
scores_npy  = np.load(DATA / "ga_scores.npy",  allow_pickle=True)
chromos_npy = np.load(DATA / "ga_chromos.npy", allow_pickle=True)

# 1) population -> CSV (each chromosome -> row of 0/1)
pop_rows = [[1 if x else 0 for x in mask] for mask in pop_npy]
pd.DataFrame(pop_rows).to_csv(DATA / "ga_pop.csv", index=False)

# 2) scores -> CSV
pd.DataFrame({"best_score": scores_npy}).to_csv(DATA / "ga_scores.csv", index=False)

# 3) best-over-all -> extra small CSV
best_idx = int(np.argmax(scores_npy))
best_mask = chromos_npy[best_idx]
best_cols = np.where(best_mask)[0]

pd.DataFrame({
    "best_gen_idx": [best_idx],
    "best_acc": [float(scores_npy[best_idx])],
    "n_selected": [len(best_cols)],
    "selected_cols": [",".join(map(str, best_cols))]
}).to_csv(DATA / "ga_best.csv", index=False)

print(f"saved to: {DATA}")
