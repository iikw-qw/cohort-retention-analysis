import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    signups, events = data.generate_events()
    activity = core.activity_table(signups, events)
    matrix = core.retention_matrix(activity, signups)
    last = core.to_month(events["event_date"]).max()
    matrix = core.mask_future(matrix, last)
    curve = core.average_curve(matrix)
    m1 = core.cohort_quality(matrix, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
    im = axes[0].imshow(matrix.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(matrix.shape[1]))
    axes[0].set_yticks(range(matrix.shape[0]))
    axes[0].set_yticklabels([str(i) for i in matrix.index])
    axes[0].set_xlabel("months since signup")
    axes[0].set_title("Cohort retention")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v:.0%}", ha="center", va="center",
                             color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=axes[0])
    axes[1].plot(curve.index, curve.values, marker="o", color="black")
    axes[1].set_title("Average retention curve")
    axes[1].set_xlabel("months since signup")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "retention.png", dpi=130)

    trend = np.polyfit(np.arange(len(m1)), m1.values, 1)[0]
    summary = {
        "users": int(len(signups)),
        "cohorts": int(matrix.shape[0]),
        "avg_month1_retention": round(float(curve.get(1, np.nan)), 3),
        "avg_month3_retention": round(float(curve.get(3, np.nan)), 3),
        "month1_retention_trend_per_cohort": round(float(trend), 4),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    matrix.round(4).to_csv(OUT / "retention_matrix.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
