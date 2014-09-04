import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import hyperopt
from IPython import display

def lossof(x):
    try:
        return float(x)
    except:
        return np.inf

def scatter_error_vs_time(estimator, ax):
    losses = estimator.trials.losses()
    ax.set_title('Job Error Throughout Run')
    ax.set_ylabel('Validation error rate')
    ax.set_xlabel('Iteration')
    ax.scatter(range(len(losses)), losses)


def plot_minvalid_vs_time(estimator, ax, ylim=None):
    losses = map(lossof, estimator.trials.losses())
    ts = range(1, len(losses))
    mins = [np.min(losses[:ii]) for ii in ts]
    ax.set_ylabel('Validation error)')
    ax.set_xlabel('Iteration')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title('Min Loss to Date')
    ax.plot(ts, mins)


def plot_duration_vs_time(estimator, ax, ylim=None):
    def duration_of(tr):
        delta = (tr['refresh_time'] - tr['book_time'])
        return delta.total_seconds()
    durations = map(duration_of, estimator.trials.trials)
    ax.set_ylabel('Seconds')
    ax.set_xlabel('Iteration')
    ax.set_title('Job duration')
    ax.scatter(range(len(durations)), durations)


class PlotHelper(object):
    def __init__(self, estimator, mintodate_ylim=None, figsize=(16, 3.5)):
        self.estimator = estimator
        self.fig, self.axs = plt.subplots(1, 3, figsize=figsize)
        self.post_iter_wait = .3
        self.mintodate_ylim = mintodate_ylim
        self.t0 = time.time()

    def post_iter(self):
        self.axs[0].clear()
        self.axs[1].clear()
        scatter_error_vs_time(self.estimator, self.axs[0])
        plot_minvalid_vs_time(self.estimator, self.axs[1],
                              ylim=self.mintodate_ylim)
        plot_duration_vs_time(self.estimator, self.axs[2])
        self.post_loop()
        #display.clear_output()
        display.display(self.fig)
        now = datetime.datetime.now()
        display.display('Last update: %s' % (
            now.strftime('%H:%M:%S %b %d, %Y')))
        time.sleep(self.post_iter_wait)

    def post_loop(self):
        display.clear_output()
        print('Total trials: %s' % len(self.estimator.trials.trials))
        print('Successful trials: %s' % len(
            filter(lambda st: st == hyperopt.STATUS_OK,
                   self.estimator.trials.statuses())))
        print('Failed trials: %s' % len(
            filter(lambda st: st != hyperopt.STATUS_OK,
                   self.estimator.trials.statuses())))
        losses = map(lossof, self.estimator.trials.losses())
        print('Best validation error: %s' % min(losses))

        print('Total wall time: %.1f minutes' % ((time.time() - self.t0) / 60.))
