import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16

environments = ("HalfCheetah", "Hopper", "Walker2D")
penguin_means = {
    'MOPO': (35.9, 16.7, 4.2),
    'TATU+MOPO': (39.3, 31.9, 10.4),
    'NUNO': (52.3, 73.5, 27.1),
}

x = np.arange(len(environments))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(8,4), layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, edgecolor='black',)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Human-normalized score')
# ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, environments)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)

plt.savefig("comparison_lowquality.pdf", dpi=1000)