from __future__ import division, print_function

import numpy as np
from scipy.interpolate import interp1d

from discrete_family import discrete_family
from barrier_affine import solve_barrier_tree

def _approx_log_reference(query_spec,
                          observed_target,
                          linear_coef,
                          grid):
    # Actual parameters to be passed: node, eval_grid, nuisance
    """
    Approximate the log of the reference density on a grid.
    """
    ## TODO: 1. reconstruct Q from the grid
    ## TODO: 2. Perform Laplace approximation for each grid, and for each node split
    ## TODO: 3. Add back the constant term omitted in Laplace Approximation
    ## TODO: 4. Return reference measure


    QS = query_spec
    cond_precision = np.linalg.inv(QS.cond_cov)

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    ref_hat = []
    solver = solve_barrier_tree
    for k in range(grid.shape[0]):
        cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) + QS.cond_mean)
        conjugate_arg = cond_precision.dot(cond_mean_grid)

        val, _, _ = solver(conjugate_arg = conjugate_arg,
                           precision = cond_precision,
                           feasible_point = QS.observed_soln,
                           con_linear = QS.linear_part,
                           con_offset = QS.offset)

        ref_hat.append(-val - (conjugate_arg.T.dot(QS.cond_cov).dot(conjugate_arg) / 2.))

    return np.asarray(ref_hat)

def _approx_log_reference(query_spec,
                          observed_target,
                          linear_coef,
                          node,
                          root,
                          X,
                          grid):
    # Actual parameters to be passed: node, eval_grid, nuisance
    """
    Approximate the log of the reference density on a grid.
    """
    ## TODO: 1. reconstruct Q from the grid
    ## TODO: 2. Perform Laplace approximation for each grid, and for each node split
    ## TODO: 3. Add back the constant term omitted in Laplace Approximation
    ## TODO: 4. Return reference measure

    ## query_spec contains
    ## 1. observed_solution for o at each split
    ## 2.

    QS = query_spec
    cond_precision = np.linalg.inv(QS.cond_cov)

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    ref_measure = []

    ref_hat = []
    solver = solve_barrier_tree
    for k in range(grid.shape[0]):
        cond_mean_grid = (linear_coef.dot(np.atleast_1d(grid[k] - observed_target)) + QS.cond_mean)
        conjugate_arg = cond_precision.dot(cond_mean_grid)

        Q = ...
        observed_soln = ...
        val, _, _ = solver(Q,
                           precision=cond_precision,
                           feasible_point = observed_soln)

        ref_hat.append(-val - (conjugate_arg.T.dot(QS.cond_cov).dot(conjugate_arg) / 2.))

    return np.asarray(ref_hat)


def _approx_log_reference(self, node, grid, nuisance,
                            contrast, norm_constrast, sd=1):
    ## TODO: 1. reconstruct Q from the grid
    ## TODO: 2. Perform Laplace approximation for each grid, and for each node split
    ## TODO: 3. Add back the constant term omitted in Laplace Approximation
    ## TODO: 4. Return reference measure

    prev_branch = node.prev_branch
    current_depth = node.depth
    node = self.root
    ## TODO: Move the node according to branch when evaluating integrals
    # Subsetting the covariates to this current node
    X = self.X[node.membership.astype(bool)]
    ref_hat = []
    norm = np.linalg.norm(contrast)
    depth = 0

    while depth <= current_depth:
        for g in grid:
            y_g = g * norm * sd + nuisance
            # TODO: Account for depth here

            y_left = self.y[node.left.membership.astype(bool)]
            y_right = self.y[node.left.membership.astype(bool)]
            optimal_loss = self._calculate_loss(y_left, y_right,
                                                randomization=0)
            randomization = node.randomization
            S_total, J_total = randomization.shape
            implied_mean = []

            # TODO: Add a layer to account for depth of the tree
            for j in range(J_total):
                feature_values = X[:, j]
                feature_values_sorted = feature_values.copy()
                feature_values_sorted.sort()
                for s in range(S_total - 1):
                    threshold = feature_values_sorted[s]
                    X_left, y_left, X_right, y_right \
                        = self._split(X, y_g, j, threshold)
                    implied_mean_s_j = self._calculate_loss(y_left, y_right,
                                                            randomization=0)
                    implied_mean.append(implied_mean_s_j)

                    # The implied mean is given by the optimal loss minus
            # the loss at each split
            # implied_mean: Q(eta'Y; nuisance)
            implied_mean = np.array(implied_mean)
            implied_mean = optimal_loss - implied_mean
            n_opt = len(implied_mean)
            implied_cov = np.ones((n_opt, n_opt)) + np.eye(n_opt)
            # sel_prob = mvn(mean=implied_mean, cov=implied_cov).cdf(np.zeros(n_opt))

            #ref_hat.append(np.log(sel_prob))

        # TODO: Move to the next layer
        depth += 1
        node = ... # Depend on where the branch demands

    return np.array(ref_hat)

def split_inference(y, node, sd=1,
                    ngrid=10000, ncoarse=100,
                    level=0.9):
    """
    Inference for a split of a node
    :param y: response vector (n x 1)
    :param node: the node whose split is of interest
    :return: p-values for difference in mean
    """
    # First determine the projection direction
    left_membership = node.left.membership
    right_membership = node.right.membership
    contrast = left_membership / np.sum(left_membership) - right_membership / np.sum(right_membership)
    norm_constrast = contrast / (np.linalg.norm(contrast) * sd)

    # Using the normalized contrast in practice
    # for scale-free grid approximation
    observed_target = norm_constrast @ y
    # The nuisance parameter is defined the same way
    nuisance = (y - np.linalg.outer(contrast, contrast)
                @ y / (np.linalg.norm(contrast) ** 2))

    stat_grid = np.linspace(-10, 10, num=ngrid)

    if ncoarse is not None:
        coarse_grid = np.linspace(-10, 10, ncoarse)
        eval_grid = coarse_grid
    else:
        eval_grid = stat_grid

    ## TODO: Rescale the statistic to have unit variance
    ## TODO: Make the grid approximating the distribution of a scale-free quantity
    # Evaluate reference measure (selection prob.) over stat_grid
    ref = _approx_log_reference(node, eval_grid, nuisance)

    if ncoarse is None:
        logWeights = np.zeros((ngrid,))
        for g in range(ngrid):
            # Evaluate the log pdf as a sum of (log) gaussian pdf
            # and (log) reference measure
            # TODO: Check if the original exp. fam. density is correct
            #logWeights[g] = ( - 0.5 * (stat_grid[g] - observed_target) ** 2
            #                  / np.linalg.norm(contrast) ** 2 + ref[g] )
            logWeights[g] = (- 0.5 * (stat_grid[g]) ** 2
                             / np.linalg.norm(contrast) ** 2 + ref[g] )
        # normalize logWeights
        logWeights = logWeights - np.max(logWeights)
        condl_density = discrete_family(eval_grid,
                                        np.exp(logWeights),
                                        logweights=logWeights)
    else:
        # print("Coarse grid")
        approx_fn = interp1d(eval_grid,
                             ref,
                             kind='quadratic',
                             bounds_error=False,
                             fill_value='extrapolate')
        grid = np.linspace(-10, 10, num=ngrid)
        logWeights = np.zeros((ngrid,))
        for g in range(ngrid):
            # TODO: Check if the original exp. fam. density is correct
            logWeights[g] = (
                    - 0.5 * (grid[g] - observed_target) ** 2 / np.linalg.norm(contrast) ** 2 + approx_fn(grid[g]))

        # normalize logWeights
        logWeights = logWeights - np.max(logWeights)

        condl_density = discrete_family(grid, np.exp(logWeights),
                                        logweights=logWeights)

    if np.isnan(logWeights).sum() != 0:
        print("logWeights contains nan")
    elif (logWeights == np.inf).sum() != 0:
        print("logWeights contains inf")
    elif (np.asarray(ref) == np.inf).sum() != 0:
        print("ref contains inf")
    elif (np.asarray(ref) == -np.inf).sum() != 0:
        print("ref contains -inf")
    elif np.isnan(np.asarray(ref)).sum() != 0:
        print("ref contains nan")

    ## TODO: omit interval calculations
    """
    interval = (condl_density.equal_tailed_interval
                (observed=contrast.T @ y,
                 alpha=1 - level))
    if np.isnan(interval[0]) or np.isnan(interval[1]):
        print("Failed to construct intervals: nan")
    """

    pivot = condl_density.ccdf(theta=0)

    return pivot#, interval[0], interval[1]
def approx_inference(j0k0, query_spec, X_n, n, p, ngrid=10000, ncoarse=None, level=0.9):
    j0 = j0k0[0]
    k0 = j0k0[1]
    # X_n: X / sqrt(n)
    S = X_n.T @ X_n

    inner_prod = S[j0,k0] # S = X.T X / n

    S_copy = np.copy(S)
    stat_grid = np.linspace(-10, 10, num=ngrid)

    if ncoarse is not None:
        coarse_grid = np.linspace(-10, 10, ncoarse)
        eval_grid = coarse_grid
    else:
        eval_grid = stat_grid
    ref_hat = _approx_log_reference(query_spec, eval_grid, j0, k0, S_copy, n, p)
    #print("ref_hat shape:", ref_hat.shape)

    if ncoarse is None:
        logWeights = np.zeros((ngrid,))
        for g in range(ngrid):
            #print(logWeights[g])
            #print(log_det_S_j_k(eval_grid[g]))
            #print(ref_hat[g])
            logWeights[g] = log_det_S_j_k(S_copy=S_copy, j0=j0, k0=k0,
                                          s_val = eval_grid[g], n=n, p=p) + ref_hat[g]

        # plt.plot(eval_grid, logWeights)

        # normalize logWeights
        logWeights = logWeights - np.max(logWeights)
        # Set extremely small values (< e^-500) to e^-500 for numerical stability
        # logWeights_zero = (logWeights < -500)
        # logWeights[logWeights_zero] = -500
        condlWishart = discrete_family(eval_grid,
                                       np.exp(logWeights),
                                       logweights=logWeights)
    else:
        # print("Coarse grid")
        approx_fn = interp1d(eval_grid,
                             ref_hat,
                             kind='quadratic',
                             bounds_error=False,
                             fill_value='extrapolate')
        grid = np.linspace(-10, 10, num=ngrid)
        logWeights = np.zeros((ngrid,))
        for g in range(ngrid):
            #print(log_det_S_j_k(grid[g]))
            #print(approx_fn(grid[g]))
            logWeights[g] = log_det_S_j_k(S_copy=S_copy, j0=j0, k0=k0, s_val = grid[g],
                                          n=n, p=p) + approx_fn(grid[g])

        # plt.plot(grid, logWeights)

        # normalize logWeights
        logWeights = logWeights - np.max(logWeights)
        # Set extremely small values (< e^-500) to e^-500 for numerical stability
        # logWeights_zero = (logWeights < -500)
        # logWeights[logWeights_zero] = -500
        condlWishart = discrete_family(grid, np.exp(logWeights),
                                       logweights=logWeights)

    if np.isnan(logWeights).sum() != 0:
        print("logWeights contains nan")
    elif (logWeights == np.inf).sum() != 0:
        print("logWeights contains inf")
    elif (np.asarray(ref_hat) == np.inf).sum() != 0:
        print("ref_hat contains inf")
    elif (np.asarray(ref_hat) == -np.inf).sum() != 0:
        print("ref_hat contains -inf")
    elif np.isnan(np.asarray(ref_hat)).sum() != 0:
        print("ref_hat contains nan")

    neg_interval = condlWishart.equal_tailed_interval(observed=inner_prod,
                                                      alpha=1-level)
    if np.isnan(neg_interval[0]) or np.isnan(neg_interval[1]):
        print("Failed to construct intervals: nan")

    interval = invert_interval(neg_interval)

    pivot = condlWishart.ccdf(theta=0)

    return pivot, interval[0], interval[1]#neg_interval, condlWishart