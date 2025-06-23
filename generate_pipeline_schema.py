import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12,2))

# Boîtes
boxes = [
    (0, 'Collecte\nDonnées\n(API, CSV)'),
    (2.5, 'Indicateurs\nTech.\n(SMA, RSI, MACD)'),
    (5, 'Scoring &\nDivergences'),
    (7.5, 'Backtesting\n& ML'),
    (10, 'Dashboard\nStreamlit')
]

for x, label in boxes:
    rect = patches.FancyBboxPatch((x, 0), 2, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='#e0e7ef')
    ax.add_patch(rect)
    ax.text(x+1, 0.5, label, ha='center', va='center', fontsize=12)

# Flèches
for i in range(len(boxes)-1):
    ax.annotate('', xy=(boxes[i+1][0],0.5), xytext=(boxes[i][0]+2,0.5),
                arrowprops=dict(arrowstyle='->', lw=2))

ax.set_xlim(-0.5, 12)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')
plt.tight_layout()
plt.savefig('pipeline_schema.png', dpi=200)
plt.close()
