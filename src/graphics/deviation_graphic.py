import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("../../models/results_estadisticos", exist_ok=True)

# DATOS
versions = ['V1', 'V2', 'V3', 'V4', 'V5']

gru_mean  = np.array([3.99, 4.17, 4.57, 4.36, 3.33])
gru_sigma = np.array([0.46, 0.62, 0.59, 0.70, 0.22])
gru_min   = np.array([3.11, 3.09, 3.49, 3.03, 2.97])
gru_max   = np.array([5.01, 5.24, 5.65, 5.69, 3.68])
gru_amp   = gru_max - gru_min  # [1.90, 2.15, 2.16, 2.66, 0.71]

lstm_mean  = np.array([4.50, 4.25, 4.60, 4.33, 4.32])
lstm_sigma = np.array([0.70, 0.68, 0.55, 0.62, 0.63])

x = np.arange(len(versions))

COLOR_GRU        = '#7F77DD'   # purple-400
COLOR_GRU_BAND   = '#AFA9EC'   # purple-200
COLOR_LSTM       = '#1D9E75'   # teal-400
COLOR_LSTM_BAND  = '#9FE1CB'   # teal-100
COLOR_HIGHLIGHT  = '#534AB7'   # purple-600

# FIGURA 1 — MAPE medio con barras de error (σ) y rango completo
fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(x, gru_min, gru_max,
                alpha=0.15, color=COLOR_GRU_BAND, label='Rango GRU (mejor–peor)')

# Rango completo GRU (bordes punteados)
ax.plot(x, gru_min, linestyle=':', color=COLOR_GRU_BAND, linewidth=1.2)
ax.plot(x, gru_max, linestyle=':', color=COLOR_GRU_BAND, linewidth=1.2)

# Rango LSTM (±σ, banda ligera)
ax.fill_between(x, lstm_mean - lstm_sigma, lstm_mean + lstm_sigma,
                alpha=0.12, color=COLOR_LSTM_BAND)

ax.plot(x, gru_mean, 'o-', color=COLOR_GRU, linewidth=2.2, markersize=7,
        markerfacecolor=COLOR_GRU, label='GRU — MAPE medio')
ax.plot(x, lstm_mean, '^--', color=COLOR_LSTM, linewidth=2.2, markersize=7,
        markerfacecolor=COLOR_LSTM, label='LSTM — MAPE medio')

ax.errorbar(x, gru_mean, yerr=gru_sigma,
            fmt='none', color=COLOR_GRU, capsize=6, capthick=1.5, linewidth=1.5)
ax.errorbar(x, lstm_mean, yerr=lstm_sigma,
            fmt='none', color=COLOR_LSTM, capsize=6, capthick=1.5, linewidth=1.5)

# Etiquetas de valor sobre cada punto GRU
for i, (m, s) in enumerate(zip(gru_mean, gru_sigma)):
    ax.annotate(f'{m:.2f}%\n±{s:.2f}',
                xy=(x[i], m), xytext=(0, 14),
                textcoords='offset points', ha='center', fontsize=8.5,
                color=COLOR_HIGHLIGHT, fontweight='500')

# Etiquetas de valor sobre cada punto LSTM
for i, (m, s) in enumerate(zip(lstm_mean, lstm_sigma)):
    ax.annotate(f'{m:.2f}%\n±{s:.2f}',
                xy=(x[i], m), xytext=(0, -28),
                textcoords='offset points', ha='center', fontsize=8.5,
                color='#0F6E56', fontweight='500')

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=11)
ax.set_ylabel('MAPE (%)', fontsize=11)
ax.set_ylim(2.2, 6.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.grid(axis='x', visible=False)
ax.spines[['top', 'right']].set_visible(False)

ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_title('MAPE medio ± σ por versión — GRU vs LSTM (20 semillas)',
             fontsize=13, fontweight='500', pad=14)

plt.tight_layout()
path1 = "../../models/results_estadisticos/mape_sigma_versiones.png"
plt.savefig(path1, dpi=300)
plt.close()
print(f"Guardado: {path1}")

# FIGURA 2 — Amplitud del rango GRU (mejor–peor, en pp)
fig, ax = plt.subplots(figsize=(8, 4.5))

bar_colors = [COLOR_GRU_BAND] * 4 + [COLOR_GRU]  # V5 destacada
bars = ax.bar(x, gru_amp, color=bar_colors, width=0.55,
              edgecolor='white', linewidth=0.8)

for bar, amp in zip(bars, gru_amp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f'{amp:.2f} pp', ha='center', va='bottom', fontsize=9.5, fontweight='500',
            color=COLOR_HIGHLIGHT)

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=11)
ax.set_ylabel('Amplitud (pp)', fontsize=11)
ax.set_ylim(0, 3.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f} pp'))
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.grid(axis='x', visible=False)
ax.spines[['top', 'right']].set_visible(False)

highlight_patch = mpatches.Patch(color=COLOR_GRU,    label='V5 — mín. amplitud (0.71 pp)')
base_patch      = mpatches.Patch(color=COLOR_GRU_BAND, label='V1–V4')
ax.legend(handles=[base_patch, highlight_patch], fontsize=9, framealpha=0.9)

ax.set_title('Amplitud del rango GRU por versión (mejor – peor semilla)',
             fontsize=13, fontweight='500', pad=14)

plt.tight_layout()
path2 = "../../models/results_estadisticos/amplitud_rango_gru.png"
plt.savefig(path2, dpi=300)
plt.close()
print(f"Guardado: {path2}")

# FIGURA 3 — σ GRU vs σ LSTM comparados en barras agrupadas
fig, ax = plt.subplots(figsize=(9, 5))

width = 0.35
bars_gru  = ax.bar(x - width/2, gru_sigma,  width, label='σ GRU',  color=COLOR_GRU,
                   edgecolor='white', linewidth=0.8)
bars_lstm = ax.bar(x + width/2, lstm_sigma, width, label='σ LSTM', color=COLOR_LSTM,
                   edgecolor='white', linewidth=0.8)

for bar, val in zip(bars_gru, gru_sigma):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'±{val:.2f}', ha='center', va='bottom', fontsize=8.5,
            color=COLOR_HIGHLIGHT, fontweight='500')

for bar, val in zip(bars_lstm, lstm_sigma):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'±{val:.2f}', ha='center', va='bottom', fontsize=8.5,
            color='#0F6E56', fontweight='500')

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=11)
ax.set_ylabel('Desviación típica σ (%)', fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}%'))
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.grid(axis='x', visible=False)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_title('Desviación típica (σ) del MAPE — GRU vs LSTM por versión',
             fontsize=13, fontweight='500', pad=14)

plt.tight_layout()
path3 = "../../models/results_estadisticos/sigma_gru_vs_lstm.png"
plt.savefig(path3, dpi=300)
plt.close()
print(f"Guardado: {path3}")


# GUARDAR RESUMEN
with open("../../models/results_estadisticos/resumen_estadistico.txt", "w") as f:
    f.write("RESUMEN ESTADÍSTICO — GRU vs LSTM (20 semillas por versión)\n")
    f.write("=" * 62 + "\n\n")
    f.write(f"{'Versión':<10} {'σ GRU':>8} {'σ LSTM':>8} {'Mejor GRU':>12} {'Peor GRU':>10} {'Amplitud':>10}\n")
    f.write("-" * 62 + "\n")
    for i, v in enumerate(versions):
        f.write(f"{v:<10} {gru_sigma[i]:>7.2f}% {lstm_sigma[i]:>7.2f}% "
                f"{gru_min[i]:>11.2f}% {gru_max[i]:>9.2f}% {gru_amp[i]:>9.2f} pp\n")
    f.write("\n")
    f.write(f"GRU  más estable: V5 (σ = {gru_sigma[4]:.2f}%, amplitud = {gru_amp[4]:.2f} pp)\n")
    f.write(f"GRU  más preciso (media): V5 ({gru_mean[4]:.2f}%)\n")
    f.write(f"LSTM más estable: V3 (σ = {lstm_sigma[2]:.2f}%)\n")

print("Resumen guardado en resumen_estadistico.txt")
print("\nFicheros generados:")
print(f"  {path1}")
print(f"  {path2}")
print(f"  {path3}")