import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

os.makedirs("../../models/results_estadisticos", exist_ok=True)

# PEORES PREDICCIONES MEDIAS
data = [
    ("2023-12-06 07:00",  916.5, 307.0, 18),
    ("2023-11-08 09:00", 2151.0, 298.6, 20),
    ("2023-12-25 07:00",  861.0, 278.7, 19),
    ("2023-12-06 08:00",  923.0, 263.7,  7),
    ("2023-12-21 07:00", 1657.0, 263.2,  1),
    ("2023-12-27 09:00", 1362.0, 246.8, 18),
    ("2023-12-18 07:00", 1630.0, 244.1,  7),
    ("2023-12-19 07:00", 1674.0, 242.4,  3),
    ("2023-10-30 09:00", 1655.0, 239.9, 17),
    ("2023-12-26 07:00", 1203.5, 239.6,  1),
    ("2023-11-27 07:00", 1537.0, 236.8, 16),
    ("2023-12-20 07:00", 1724.0, 236.1,  5),
    ("2023-12-18 09:00", 2665.0, 227.6, 10),
    ("2023-11-02 09:00", 1751.0, 225.2, 19),
    ("2023-10-23 06:00", 1139.0, 222.8,  4),
]

data_sorted  = sorted(data, key=lambda x: x[3], reverse=True)
labels       = [d[0][5:16] for d in data_sorted]   # "12-06 07:00"
occurrences  = np.array([d[3] for d in data_sorted])
errors       = np.array([d[2] for d in data_sorted])

COLOR_SYSTEMATIC = "#7F77DD"
COLOR_FREQUENT   = "#AFA9EC"
COLOR_SPORADIC   = "#CECBF6"

def assign_color(oc):
    if oc >= 15: return COLOR_SYSTEMATIC
    if oc >= 8:  return COLOR_FREQUENT
    return COLOR_SPORADIC

bar_colors = [assign_color(o) for o in occurrences]

# GRÁFICA
fig, ax = plt.subplots(figsize=(14, 7))

x_pos = np.arange(len(labels))
bars  = ax.bar(x_pos, occurrences, width=0.62,
               color=bar_colors, edgecolor="white", linewidth=0.6)

ax.axhline(y=20, color="#888780", linestyle="--", linewidth=1,
           alpha=0.6, label="Maximum possible (20 seeds)")
ax.axhline(y=15, color=COLOR_SYSTEMATIC, linestyle=":", linewidth=1.2,
           alpha=0.7, label="Systematic threshold (15 seeds)")

for bar, err in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{err:.0f} Wh",
            va="bottom", ha="center",
            fontsize=8, color="#5F5E5A")

legend_patches = [
    Patch(color=COLOR_SYSTEMATIC, label="Systematic  ≥15 seeds"),
    Patch(color=COLOR_FREQUENT,   label="Frequent    8–14 seeds"),
    Patch(color=COLOR_SPORADIC,   label="Sporadic    <8 seeds"),
]
ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9, loc="upper right")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Occurrences in Top 15 (out of 20 seeds)", fontsize=10)
ax.set_ylim(0, 23)
ax.set_title("Recurrence of worst errors — GRU V5 (20 seeds)",
             fontsize=13, fontweight="500", pad=12)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out_path = "../../models/results_estadisticos/apariciones_top15_v5.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")