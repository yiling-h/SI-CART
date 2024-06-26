from __future__ import division, print_function

import numpy as np
from scipy.interpolate import interp1d

from discrete_family import discrete_family
from barrier_affine import solve_barrier_affine_py


def _approx_log_reference(observed_target,
                          cov_target,
                          linear_coef,
                          grid):
    """
    Approximate the log of the reference density on a grid.
    """

    QS = self.query_spec
    cond_precision = np.linalg.inv(QS.cond_cov)

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    ref_hat = []
    solver = solve_barrier_affine_py

    for k in range(grid.shape[0]):
        cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) + QS.cond_mean)
        conjugate_arg = cond_precision.dot(cond_mean_grid)

        val, _, _ = solver(conjugate_arg,
                           cond_precision,
                           QS.observed_soln,
                           QS.linear_part,
                           QS.offset,
                           **self.solve_args)

        ref_hat.append(-val - (conjugate_arg.T.dot(QS.cond_cov).dot(conjugate_arg) / 2.))

    return np.asarray(ref_hat)