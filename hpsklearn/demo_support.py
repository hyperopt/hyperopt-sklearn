import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def scatter_error_vs_time(estimator, ax):
    losses = estimator.trials.losses()
    ax.set_ylabel('Validation error rate')
    ax.set_xlabel('Iteration')
    ax.scatter(range(len(losses)), losses)


def plot_minvalid_vs_time(estimator, ax, ylim=None):
    losses = estimator.trials.losses()
    mins = [np.min(losses[:ii]) for ii in range(1, len(losses))]
    ax.set_ylabel('min(Validation error rate to-date)')
    ax.set_xlabel('Iteration')
    if ylim:
        plt.ylim(*ylim)
    plt.plot(mins)


def iris_plot_per_iter(estimator):
    display.clear_output()
    fig, axs = plt.subplots(1, 2)
    scatter_error_vs_time(estimator, axs[0])
    plot_minvalid_vs_time(estimator, axs[1], ylim=(-.01, .05))
    display.display(fig)
    time.sleep(0.5)

