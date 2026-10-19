import numpy as np

from .metrics import bin_index, reliability_bins


def prediction_histogram(ax, p, color, n_bins=10):
    counts = np.bincount(bin_index(p, n_bins), minlength=n_bins)
    ax.bar(np.arange(n_bins) / n_bins, counts, width=1 / n_bins, align="edge",
           color=color, edgecolor="white", linewidth=0.8)
    ax.set(xlim=(0, 1), xticks=np.linspace(0, 1, 6), xlabel="Predicted probability",
           ylabel="Count")


def reliability_diagram(ax, y, p, color, label, n_bins=10, connect=True, curves=()):
    mean_pred, frac_pos, _ = reliability_bins(y, p, n_bins)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for x, f, c, curve_label in curves:
        ax.plot(x, f, color=c, linewidth=1.5, label=curve_label)
    ax.plot(mean_pred, frac_pos, marker="o", linestyle="-" if connect else "none",
            color=color, markersize=4, label=label)
    ticks = np.linspace(0, 1, 6)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=ticks, yticks=ticks, box_aspect=1,
           xlabel="Mean predicted probability", ylabel="Fraction of positives")

def calibration_size_curve(ax, sizes, series, uncalibrated, uncalibrated_color):
    ax.axhline(uncalibrated, color=uncalibrated_color, linestyle="--", label="Uncalibrated")
    for y, color, label in series:
        ax.plot(sizes, y, marker="o", markersize=3.5, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(sizes, labels=[str(s) for s in sizes])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelrotation=45)
    
def reliability_axes(ax, diagonal_label=None, xlabel="Mean predicted probability",
                     ylabel="Fraction of positives"):
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label=diagonal_label)
    ticks = np.linspace(0, 1, 6)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=ticks, yticks=ticks, box_aspect=1,
           xlabel=xlabel, ylabel=ylabel)
    
def reliability_curve(ax, y, p, n_bins=10, **style):
    mean_pred, frac_pos, _ = reliability_bins(y, p, n_bins)
    ax.plot(mean_pred, frac_pos, **style)