import os
import numpy as np
from contextlib import contextmanager
import time
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

NAN = -100000


def wrap(x):
    """
    Wrapping operator on real data.
    """
    return np.angle(np.exp(1j * x))


@contextmanager
def no_print():
    old = sys.stdout
    fileobj = open('printed.txt', 'w')
    # sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old
        os.remove('printed.txt')


@contextmanager
def timer(op_string, will_print=False):
    global temp_time
    start_time = -time.time()
    fileobj = open('printed.txt', 'w')
    try:
        yield fileobj
    finally:
        total_time = 1000 * (time.time() + start_time)
        if will_print: print("{0:.<40}{1:.>15}".format('[TIME: ' + op_string + ']',
                                                       ' {:4.2f}'.format(total_time) + ' ms'))
        temp_time = total_time


def fitness_calc(sol, probs):
    npt, arcs, arc_cost, arc_est, ptid, pt_cost, pt_est = probs
    s, t = arcs.T
    sol = sol.reshape(16, 16)
    sol = np.cumsum(sol, axis=0)
    sol = sol.reshape(-1)
    pairwise_terms = np.sum(arc_cost * np.abs(sol[t] - sol[s] + arc_est))
    energy = pairwise_terms
    return energy


def plot_figure(found_labels_r, labels_gt, ifg_whole, label=False):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    m, n = ifg_whole.shape
    found_labels_r = found_labels_r.reshape(m, n)
    found_labels_r = np.cumsum(found_labels_r, axis=0)

    im0 = ax0.imshow(found_labels_r.reshape(m, n).T, interpolation='nearest',
                     cmap='jet')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im0, cax=cax0)
    ax0.set_title('Our Labels')

    error = (found_labels_r.reshape(m, n) - labels_gt)
    im3 = ax3.imshow(error.T, interpolation='nearest',
                     cmap='jet')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im3, cax=cax3)
    ax3.set_title('Error: [' + str(np.min(error)) + ', ' + str(np.max(error)) + ']')

    im1 = ax1.imshow(labels_gt.T, interpolation='nearest',
                     cmap='jet')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax1.set_title('Ground Truth: [' + str(np.min(labels_gt)) + ', ' + str(np.max(labels_gt)) + ']')

    im2 = ax2.imshow(ifg_whole.reshape(m, n).T, interpolation='nearest',
                     cmap='jet', vmax=np.pi, vmin=-np.pi)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax2.set_title('Wrapped Input')

    if label:
        toShowWrapped = ifg_whole.reshape(m, n).T
        for i in range(toShowWrapped.shape[0]):
            for j in range(toShowWrapped.shape[1]):
                ax2.text(j, i, '{:4.1f}'.format(toShowWrapped[i][j]), ha="center", va="center", color="k")

        ax0_value = found_labels_r.reshape(m, n)
        for i in range(ax0_value.shape[0]):
            for j in range(ax0_value.shape[1]):
                ax0.text(i, j, '{:2.0f}'.format(ax0_value[i][j]), ha="center", va="center", color="k")

        ax3_value = (found_labels_r.reshape(m, n) - labels_gt)
        for i in range(ax3_value.shape[0]):
            for j in range(ax3_value.shape[1]):
                ax3.text(i, j, '{:2.0f}'.format(ax3_value[i][j]), ha="center", va="center", color="k")

        for i in range(labels_gt.shape[0]):
            for j in range(labels_gt.shape[1]):
                ax1.text(i, j, '{:2.0f}'.format(labels_gt[i][j]), ha="center", va="center", color="k")
    return fig


def probs_gen(ifg):
    # Generate whole graph
    reglambda = 100
    m, n = ifg.shape
    ngb_rad = 1
    arcs = set()
    for y in range(m):
        for x in range(n):
            for d0 in range(-ngb_rad, ngb_rad + 1):
                for d1 in range(-ngb_rad, ngb_rad + 1):
                    if d0 == 0 and d1 == 0:
                        continue
                    if d0 != 0 and d1 != 0:
                        continue
                    nx, ny = x + d1, y + d0
                    if nx < 0 or ny < 0 or nx >= n or ny >= m:
                        continue  # invalid
                    idx = y * n + x
                    nidx = ny * n + nx
                    arc = tuple(sorted((idx, nidx)))
                    arcs.add(arc)

    arcs = np.asarray(sorted(list(arcs)))
    s, t = arcs.T
    ifgr = ifg.reshape(-1)
    darc = ifgr[t] - ifgr[s]
    arc_est = np.round((wrap(darc) - darc) / (2 * np.pi)).astype('int32')
    arc_cost = reglambda * np.ones(arc_est.shape[0], dtype='int32')
    m, n = ifg.shape
    pt_est = np.zeros(m * n, dtype='int32')
    pt_cost = 1 * np.ones(m * n, dtype='int32')  # Assigning zero means no unary cost
    pt_idx = np.arange(m * n)
    npt = m * n

    probs = npt, arcs, arc_cost, arc_est, pt_idx, pt_cost, pt_est
    return probs
