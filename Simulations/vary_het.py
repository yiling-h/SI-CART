"""Smooth-mean, heteroskedastic Gaussian simulation with a working common SD.

Run from Simulations, using the same positional interface as vary_signal.py:
    python3 vary_het.py 0 5 '[0,0.5,1]' het

Each worker checkpoints to <prefix>_<start>_<end>.pkl after each replication.
The results dictionary retains vary_signal's columns, adds ``h``, and records
the estimated working SD (not h or the true noise SD) in ``SD(Y)``. ``Length``
is interval width times sqrt(leaf size), as in vary_signal.

Inference and tree settings are reused from vary_signal, including its R and
treevalues dependencies. All methods assume a common SD estimated by std(Y).
This is a robustness experiment, without an exact-coverage guarantee.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np


def generate_het_data(n=200, p=10, h=0., seed=1000):
    """Return paired design, means, training responses and test responses.

    Reusing a seed across h values reuses X and the two independent standard
    normal error vectors. Test responses are generated at the training X.
    """
    if n < 2 or p < 4:
        raise ValueError("Require n >= 2 and p >= 4")
    if not np.isfinite(h) or not 0 <= h <= 1:
        raise ValueError("h must be finite and between 0 and 1")
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    mu = X[:, 0] + .5 * X[:, 1]**2 - 2 * X[:, 2] * X[:, 3]
    # 1 + h*tanh(2*x), evaluated without cancellation near h=1, x << 0.
    # This equals 4*{1 + h*tanh(2*x)} and E[variance]=4 for every h.
    variance = 4 * ((1 - h) + 2 * h * np.exp(-np.logaddexp(0., -4 * X[:, 0])))
    sd = np.sqrt(variance)
    y = mu + sd * rng.normal(size=n)
    y_test = mu + sd * rng.normal(size=n)
    return X, mu, y, y_test


def vary_het_sim(n=200, p=10, h_list=(0., .5, 1.), noise_sd=1.,
                 start=0, end=100, level=.1, path=None):
    h_list = tuple(h_list)
    if start < 0 or end <= start:
        raise ValueError("Require 0 <= start < end")
    if (not h_list or len(set(h_list)) != len(h_list)
            or any(not np.isfinite(h) or not 0 <= h <= 1 for h in h_list)):
        raise ValueError("h_list must contain distinct finite values in [0, 1]")
    # Lazy imports allow inspecting the CLI or DGP without starting R.
    from vary_signal import (
        RegressionTree, randomized_inference, tree_values_inference,
        UV_decomposition,
    )

    results = {key: [] for key in
               ("Coverage Rate", "Length", "MSE", "Method", "SD(Y)", "h")}
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def record(method, coverage, lengths, prediction, y_test, sd_hat, h):
        results["Coverage Rate"].append(float(np.mean(coverage)))
        results["Length"].append(float(np.mean(lengths)))
        results["MSE"].append(float(np.mean((y_test - prediction)**2)))
        results["Method"].append(method)
        results["SD(Y)"].append(sd_hat)
        results["h"].append(h)

    for i in range(start, end):
        for h in h_list:
            X, mu, y, y_test = generate_het_data(n=n, p=p, h=h, seed=i + 1000)
            sd_hat = float(np.std(y, ddof=1))
            if not np.isfinite(sd_hat) or sd_hat <= 0:
                raise ValueError(f"Invalid working SD for replication {i}, h={h}")
            # Pair randomization across h, independently of DGP random draws.
            np.random.seed(i + 1000)
            print(f"Replication {i}, h={h:g}, working SD={sd_hat:.6g}", flush=True)

            reg_tree = RegressionTree(min_samples_split=50, max_depth=3,
                                      min_proportion=0., min_bucket=20)
            reg_tree.fit(X, y, sd=noise_sd * sd_hat)
            coverage, lengths = randomized_inference(
                reg_tree=reg_tree, y=y, sd_y=sd_hat, mu=mu,
                level=level, noise_sd=noise_sd, reduced_dim=None, prop=.2)
            record("RRT", coverage, lengths, reg_tree.predict(X), y_test, sd_hat, h)

            # Preserve vary_signal's Tree-Values pipeline, including its
            # rpart minsplit=25/minbucket=10 and maximum depth three.
            coverage, lengths, prediction = tree_values_inference(
                X, y, mu, sd_y=sd_hat, X_test=X, max_depth=3, level=level)
            record("Tree-Values", coverage, lengths, prediction, y_test, sd_hat, h)

            coverage, lengths, prediction = UV_decomposition(
                X, y, mu, sd_hat, X_test=X, min_prop=0., max_depth=3,
                min_sample=50, min_bucket=20, gamma=.05, level=level)
            record("UV(0.05)", coverage, lengths, prediction, y_test, sd_hat, h)

        if path is not None:
            joblib.dump(results, path)
            print(f"Checkpointed through replication {i}: {path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=int, help="First replication (inclusive)")
    parser.add_argument("end", type=int, help="Last replication (exclusive)")
    parser.add_argument("h_list", help="Heteroskedasticity levels, e.g. '[0,0.5,1]'")
    parser.add_argument("prefix", help="Output filename prefix, optionally with a directory")
    args = parser.parse_args()
    try:
        h_list = [float(x) for x in args.h_list.strip("[]").split(",") if x.strip()]
    except ValueError:
        parser.error("h_list must be a comma-separated numeric list")
    if (args.start < 0 or args.end <= args.start or not h_list
            or len(set(h_list)) != len(h_list)
            or any(not np.isfinite(h) or not 0 <= h <= 1 for h in h_list)):
        parser.error("Require 0 <= start < end and distinct h values in [0, 1]")

    # The reused helper explicitly converts NumPy arrays with numpy2ri.py2rpy.
    # Global pandas conversion is unnecessary and activate() errors in rpy2 3.6.
    from vary_signal import importr
    importr('treevalues')
    importr('rpart')

    path = f"{args.prefix}_{args.start}_{args.end}.pkl"
    vary_het_sim(n=200, p=10, h_list=h_list, noise_sd=1.,
                 start=args.start, end=args.end, level=.1, path=path)


if __name__ == '__main__':
    main()
