import pandas as pd
import matplotlib.pyplot as plt
import sys

assert len(sys.argv) > 1, "Need CSV path"

df = pd.read_csv(sys.argv[1])
df.drop(['epoch'], inplace=True, axis=1)
fig, _ = plt.subplots(1, len(df.columns), sharey=True, figsize=(3 * len(df.columns), 3), tight_layout=True)
cols = sorted(df.columns, key=lambda s: s.replace('val_', ''))
for i, c in enumerate(cols):
    fig.axes[i].plot(df[c].values)
    fig.axes[i].set_title(c)

plt.subplots_adjust(left=None, wspace=1, hspace=None, right=None)

if len(sys.argv) > 2:
    imgpath = sys.argv[2]
    plt.savefig(imgpath, dpi=300)

plt.show()
