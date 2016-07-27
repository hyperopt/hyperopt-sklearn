import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def scatter_error_vs_time(estimator, ax):
    losses = estimator.trials.losses()
    ax.set_ylabel('Validation error rate')
    ax.set_xlabel('Iteration')
    ax.scatter(list(range(len(losses))), losses)


def plot_minvalid_vs_time(estimator, ax, ylim=None):
    losses = estimator.trials.losses()
    ts = list(range(1, len(losses)))
    mins = [np.min(losses[:ii]) for ii in ts]
    ax.set_ylabel('min(Validation error rate to-date)')
    ax.set_xlabel('Iteration')
    if ylim:
        ax.set_ylim(*ylim)
    ax.plot(ts, mins)


class PlotHelper(object):

    def __init__(self, estimator, mintodate_ylim):
        self.estimator = estimator
        self.fig, self.axs = plt.subplots(1, 2)
        self.post_iter_wait = .5
        self.mintodate_ylim = mintodate_ylim

    def post_iter(self):
        self.axs[0].clear()
        self.axs[1].clear()
        scatter_error_vs_time(self.estimator, self.axs[0])
        plot_minvalid_vs_time(self.estimator, self.axs[1],
                              ylim=self.mintodate_ylim)
        display.clear_output()
        display.display(self.fig)
        time.sleep(self.post_iter_wait)

    def post_loop(self):
        display.clear_output()
