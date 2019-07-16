import numpy as np
import perlin as pe
from helper import wrap


def simulate(shape, aps_param={'corr_length': 1, 'strength': 0},
             spatial_param={'corr_length': 2, 'strength': 1.},
             baseline_param={'number': 1}, show=False, seed=None, maxambig=2):
    """
    Simulates `nlayer` SLCs of shape `m` rows x `n` cols.
    """

    if seed:
        np.random.seed(seed)

    n, m, nlayer = shape

    nbase = baseline_param['number']

    def generate_perlin_layers(param, nl):
        fudge_factor = 1.5  # empirically observed
        lcor = param['corr_length']
        lstr = param['strength']
        lyrs = list()
        for il in range(nl):
            noisegen = pe.PerlinNoiseFactory(2, seed=seed)
            lyr = fudge_factor * lstr * np.asarray([noisegen(iy, ix)
                                                    for ix in np.linspace(-lcor, lcor, n)
                                                    for iy in np.linspace(-lcor, lcor, m)]).reshape(m, n)
            lyrs.append(lyr)
        lyrs = np.asarray(lyrs)
        return lyrs

    def outer(x, y):
        return np.dot(x, y.reshape(y.shape[0], -1)).reshape((x.shape[0],) + y.shape[1:])

    aps = generate_perlin_layers(aps_param, nlayer)
    spatial_param = generate_perlin_layers(spatial_param, nbase)

    baseline = np.random.randn(nlayer, nbase)
    model = outer(baseline, spatial_param)

    total_noiseless = model + aps

    unwrapped = (total_noiseless[0] - total_noiseless[1])
    unwrapped = unwrapped - np.min(unwrapped)
    unwrapped = unwrapped / (np.max(np.abs(unwrapped)))
    unwrapped = unwrapped * (2 * np.pi * maxambig * 2)
    unwrapped = unwrapped - 2 * np.pi * maxambig

    total = unwrapped

    total = total.reshape([n, m]).T
    return wrap(total), total
