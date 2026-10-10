import numpy as np

def cv_curve(ax, values, train, val, val_std, best, baseline=None, invert=False):
    val, val_std = np.asarray(val), np.asarray(val_std)
    ax.plot(values, train, marker = "o", label="train")
    ax.plot(values, val, marker = ".", label="validation")
    ax.fill_between(values, val - val_std, val + val_std, color="C1", alpha=0.25)
    ax.axvline(best, color="red", linestyle="--", label="selected")
    if baseline is not None:
        ax.axhline(baseline, color="grey", linestyle=":", label="constant predictor")
    ax.set_xscale("log")
    ax.set_xticks(values, labels=[str(v) for v in values])
    ax.minorticks_off()
    if invert:
        ax.invert_xaxis()