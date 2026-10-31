r"""Toy-example Gaussian experiment: RRT versus Bayesian CART tree priors.

Requires Python >= 3.12 (BayesCART's requirement). Install in a separate env:
    python -m pip install \
        'bayescart @ git+https://github.com/guglielmogattiglio/bayescart.git@dd039db4b738bd981ec03dbdb96dda93b912fca4' \
        cvxpy

Only the output prefix is required:
    python Simulations/bayesianCART.py bcart

Outside SLURM this runs one replication, [0, 1). In a one-based SLURM array,
task k runs [(k-1)*50, k*50). Override the segment with --start/--end, or change
the array chunk size with --range-per-job. For example:
    python Simulations/bayesianCART.py bcart --start 0 --end 50

Data defaults match toy_example.py's entry point: n=200, p=5, sigma=2, a=1, b=2.
Each replication generates (X, mu, y, y_test) once with seed i+1000 and uses the
same observations for every method. RRT(1) is the default reference; use e.g.
--rrt-noise-sds '[0.5,1,2,5]' to run the toy example's RRT scale grid.

This writes bcart_<start>_<end>.pkl, a dictionary of parallel lists with keys
"Coverage Rate", "Length", "MSE", "Method", "SD(Y)". Length is CI width times
sqrt(leaf size), exactly as in vary_signal.py. A separate .diagnostics.pkl stores
configuration, timings, chain diagnostics, reporting-tree sizes, and versions.
Each worker checkpoints only its own start/end segment after each replication.
The output directory must exist. The Python function accepts a common path
prefix too: path='bcart.pkl' produces bcart_<start>_<end>.pkl, so distinct
workers do not overwrite each other's results or diagnostic files.

Defaults: alpha_tree=.95; beta_tree in (1, 2); independent leaf
means N(0, tau^2) with FIXED tau=10; known common observation SD. Tree depth <= 3,
min_samples_split=50, min_bucket=20. BayesCART supplies the tree representation,
grow/prune/change proposals, MCMC loop, and storage. The adapter below replaces
its unknown leaf-specific variance model and adjusts the MH ratio for our prior
and tree restrictions. Swap moves are disabled. No R or Tree-Values dependency.

For each fit, a uniformly selected retained posterior tree defines the reporting
partition. Every retained draw of the latent regression function is averaged
over each fixed reporting leaf; the 5% and 95% quantiles give the default 90% CI.
MSE uses the full-posterior mean at the training X and independent test responses,
matching the fixed-design test experiment in toy_example.py.

MCMC defaults: 5000 burn-in iterations and 2000 retained draws per independent
chain, thinning by 5, with 4 independent chains. These are starting settings,
not a guarantee of convergence. Inspect the sidecar diagnostics and increase
--burnin/--draws/--chains for production runs; rerun fits with poor diagnostics.
Each independent chain is a parallel-tempering ladder; only its cold (inverse
temperature 1) samples are retained. Heated chains temper the integrated
likelihood, leaving the tree prior unchanged. --inverse-temperatures '[1]'
disables tempering.
"""

import argparse
import os
from pathlib import Path
import sys
import warnings
from importlib.metadata import version
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

try:
    from bayescart.bcart import BCART
    from bayescart.parallel_tempering_base import BCARTPT
except ImportError as exc:
    raise ImportError(
        "This simulation needs BayesCART and Python >= 3.12. "
        "See the installation command at the top of bayesianCART.py."
    ) from exc

module_path = str(Path(__file__).resolve().parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from CART import RegressionTree


class KnownVarianceBCART(BCART):
    """BayesCART adapter for known sigma and independent N(mu_bar, tau^2) means.

    Proposals use the package's uniform variable/cut distribution, including
    cuts that violate our support; invalid proposals are rejected, not redrawn.
    The target prior chooses uniformly among variables with admissible cuts,
    then uniformly among that variable's admissible cuts. Nodes that cannot
    legally split stop with probability one. These distinctions matter for MH.
    Only continuous predictors are supported in this simulation.
    """

    def __init__(self, X, y, *, sigma, tau=10., alpha=.95, beta=2.,
                 mu_bar=0., max_depth=3, min_samples_split=50,
                 min_bucket=20, **kwargs):
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be positive and finite")
        if not 0 < alpha < 1 or not np.isfinite(beta) or beta < 0:
            raise ValueError("Require 0 < alpha < 1 and beta >= 0")
        if max_depth < 0 or min_samples_split < 2 or min_bucket < 1:
            raise ValueError("Invalid tree size constraints")
        if not np.isfinite(mu_bar):
            raise ValueError("mu_bar must be finite")
        if not np.all(np.isfinite(X.to_numpy(dtype=float))):
            raise ValueError("X must contain finite continuous predictors")
        if not np.all(np.isfinite(y.to_numpy(dtype=float))):
            raise ValueError("y must contain finite responses")
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        super().__init__(
            X, y, alpha=alpha, beta=beta, mu_bar=mu_bar,
            sigma_0=self.sigma, node_min_size=min_bucket,
            move_prob=[.3, .3, .4, 0.], **kwargs,
        )

    def sample_prior_sigma(self):
        return self.sigma

    def sample_prior_mu(self, sigma):
        return self.rng.normal(self.mu_bar, self.tau)

    def get_posterior_params_mu(self, leaf, new_sigma):
        n, mean, _ = leaf.get_data_averages()
        prior_n = self.sigma**2 / self.tau**2
        return ((n * mean + prior_n * self.mu_bar) / (n + prior_n),
                self.sigma / np.sqrt(n + prior_n))

    def resample_mu_sigma(self):
        for leaf in self.tree.get_leaves():
            mean, sd = self.get_posterior_params_mu(leaf, self.sigma)
            leaf.update_node_params((self.rng.normal(mean, sd), self.sigma))

    def _calc_llik(self, tree):
        # Integrate each N(mu_bar, tau^2) leaf mean analytically. No variance prior.
        n, mean, sse = self.get_leaves_means(tree)
        sigma2, tau2 = self.sigma**2, self.tau**2
        return float(np.sum(
            -.5 * n * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2)
            -.5 * np.log1p(n * tau2 / sigma2)
            -.5 * n * (mean - self.mu_bar)**2 / (sigma2 + n * tau2)
        ))

    def _valid_splits(self, node):
        if node.depth >= self.max_depth or node.get_nobs() < self.min_samples_split:
            return {}
        available, categorical = node.get_available_splits()
        valid = {}
        for name, cuts in available.items():
            if categorical[name]:
                raise ValueError("Only continuous predictors are supported")
            # BayesCART splits at X < cut; the complementary child has X >= cut.
            values = np.sort(node._data.X[name].to_numpy())
            sizes = np.searchsorted(values, cuts, side="left")
            keep = ((sizes >= self.node_min_size)
                    & (len(values) - sizes >= self.node_min_size))
            cuts = np.asarray(cuts)[keep]
            if cuts.size:
                valid[name] = cuts
        return valid

    def calc_log_tree_prob(self, tree):
        if tree.log_tree_prior_prob is not None:
            return tree.log_tree_prior_prob
        total = 0.
        for node in tree.all_nodes_itr():
            if node.get_nobs() < self.node_min_size:
                return -np.inf
            valid = self._valid_splits(node)
            q = self.get_p_split(node.depth) if valid else 0.
            if node.is_leaf():
                total += np.log1p(-q)
            else:
                name, cut, _ = node.get_split_info()
                if name not in valid or not np.any(valid[name] == cut):
                    return -np.inf
                total += np.log(q) - np.log(len(valid)) - np.log(len(valid[name]))
        tree.log_tree_prior_prob = total
        return total

    def calc_grow_prune_mh_data(self, new_tree, node_to_split,
                              tot_parents_with_two_leaves, b):
        # All leaves have >= min_bucket observations, so sample_leaf is uniform
        # over all b leaves. Store the proposal-rule probability before prune
        # clears this node's splitting rule. This is NOT the target rule prior.
        nvars, ncuts = node_to_split.calc_avail_split_and_vars()
        self.mh_move_data = {
            "log_reverse_forward_grow": (
                np.log(self.move_prob[1] / self.move_prob[0]) + np.log(b)
                + np.log(nvars) + np.log(ncuts)
                - np.log(tot_parents_with_two_leaves)
            )
        }

    def calc_change_swap_mh_data(self, transition_p):
        self.mh_move_data = {"log_reverse_forward_change": np.log(transition_p)}

    def trans_prob_log(self, move, new_tree, old_tree):
        prior_ratio = (self.calc_log_tree_prob(new_tree)
                       - self.calc_log_tree_prob(old_tree))
        if move == "grow":
            proposal_ratio = self.mh_move_data["log_reverse_forward_grow"]
        elif move == "prune":
            proposal_ratio = -self.mh_move_data["log_reverse_forward_grow"]
        elif move == "change":
            proposal_ratio = self.mh_move_data["log_reverse_forward_change"]
        else:
            raise ValueError("Swap proposals are disabled")
        return prior_ratio + proposal_ratio

    def get_acceptance_prob(self, new_tree, move):
        log_ratio = self.trans_prob_log(move, new_tree, self.tree)
        if np.isneginf(log_ratio):
            return 0.
        log_ratio += self.calc_llik(new_tree) - self.calc_llik(self.tree)
        if not np.isfinite(log_ratio):
            raise FloatingPointError("Non-finite Bayesian CART MH ratio")
        return float(np.exp(min(0., log_ratio)))

    def get_log_posterior_prob(self, tree):
        total = self.calc_llik(tree) + self.calc_log_tree_prob(tree)
        for leaf in tree.get_leaves():
            mean, sd = self.get_posterior_params_mu(leaf, self.sigma)
            total += (-.5 * ((leaf.get_params()[0] - mean) / sd)**2
                      - np.log(sd) - .5 * np.log(2 * np.pi))
        return total


class TemperedKnownVarianceBCART(KnownVarianceBCART, BCARTPT):
    """Use BayesCART's tempering engine with target p(T) p(y|T)^t.

    Collapsed tree moves and swaps are independent of leaf parameters. Sampling
    their ordinary full conditionals is sufficient: only t=1 draws are reported.
    """

    def get_acceptance_prob(self, new_tree, move):
        log_ratio = self.trans_prob_log(move, new_tree, self.tree)
        if np.isneginf(log_ratio):
            return 0.
        log_ratio += new_tree.temperature * (
            self.calc_llik(new_tree) - self.calc_llik(self.tree))
        if not np.isfinite(log_ratio):
            raise FloatingPointError("Non-finite tempered BCART MH ratio")
        return float(np.exp(min(0., log_ratio)))

    def get_swap_acceptance_prob(self, tree_from, tree_to):
        log_ratio = (tree_from.temperature - tree_to.temperature) * (
            self.calc_llik(tree_to) - self.calc_llik(tree_from))
        return float(np.exp(min(0., log_ratio)))


def _validate_temperatures(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) == 0 or values[0] != 1
            or not np.all(np.isfinite(values)) or np.any(values <= 0)
            or np.any(np.diff(values) >= 0)
            or (len(values) > 1 and len(values) % 2 != 0)):
        raise ValueError("Inverse temperatures must start at 1, decrease strictly "
                         "and stay positive; use one value or an even-length ladder")
    return values


def generate_test(mu, sd_y):
    n = mu.shape[0]
    return mu + np.random.normal(size=(n,), scale=sd_y)


def randomized_inference(reg_tree, sd_y, y, mu, noise_sd=1,
                         level=0.1, reduced_dim=5, prop=0.05):
    # Same computation and numerical settings as vary_signal.randomized_inference.
    coverage_i = []
    lengths_i = []
    for node in reg_tree.terminal_nodes:
        (pval, dist, contrast, norm_contrast, obs_tar, logW, suff,
         sel_probs, ref_hat_layer) = reg_tree.node_inference(
            node=node, ngrid=10000, ncoarse=300, grid_w_const=5,
            sd=sd_y, interp_kind='cubic', query_size=100)
        target = contrast.dot(mu)
        selective_CI = dist.equal_tailed_interval(
            observed=norm_contrast.dot(y), alpha=level)
        selective_CI = np.array(selective_CI)
        selective_CI *= np.linalg.norm(contrast) * sd_y
        coverage_i.append((target >= selective_CI[0] and target <= selective_CI[1]))
        lengths_i.append((selective_CI[1] - selective_CI[0]) / np.linalg.norm(contrast))
    return coverage_i, lengths_i


def tree_draw_at_X(tree, X):
    """Recover predictions and leaf memberships from a data-free stored tree."""
    predictions = np.empty(len(X))
    memberships = []

    def visit(node, indices):
        if node.is_leaf():
            predictions[indices] = node.get_params()[0]
            memberships.append(indices)
            return
        name, cut, categorical = node.get_split_info()
        if categorical:
            raise ValueError("Only continuous predictors are supported")
        left = X[name].to_numpy()[indices] < cut
        lchild, rchild = tree.get_children(node)
        visit(lchild, indices[left])
        visit(rchild, indices[~left])

    visit(tree.get_root(), np.arange(len(X)))
    return predictions, memberships


def _split_rhat(chains):
    """Basic split R-hat for scalar traces; a screening diagnostic, not proof of mixing."""
    chains = np.asarray(chains, dtype=float)
    half = chains.shape[1] // 2
    if half < 2:
        return float('nan')
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1))
    between = half * np.var(np.mean(split, axis=1), ddof=1)
    if within == 0:
        return 1. if between == 0 else float('inf')
    return float(np.sqrt(((half - 1) / half * within + between / half) / within))


def bayesian_cart_inference(X, y, mu, sd_y, *, beta=2., tau=10.,
                            alpha_tree=.95, level=.1, burnin=5000,
                            draws=2000, thin=5, chains=4, seed=0,
                            inverse_temperatures=(1., .9, .75, .55, .35, .2, .08, .03)):
    """Return leaf coverage, scaled lengths, posterior-mean predictions, diagnostics."""
    fit_start = perf_counter()
    if burnin < 1 or draws < 4 or thin < 1 or chains < 1:
        raise ValueError("Require burnin >= 1, draws >= 4, thin >= 1, chains >= 1")
    if not 0 < level < 1:
        raise ValueError("level must lie between zero and one")
    inverse_temperatures = _validate_temperatures(inverse_temperatures)
    X_frame = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    y_series = pd.Series(y, dtype=float)
    seed_sequences = np.random.SeedSequence(seed).spawn(chains + 1)
    all_predictions, all_memberships = [], []
    chain_diagnostics, leaf_traces, likelihood_traces, mean_traces = [], [], [], []
    for chain in range(chains):
        sampler_class = (KnownVarianceBCART if len(inverse_temperatures) == 1
                         else TemperedKnownVarianceBCART)
        tempering_kwargs = ({} if len(inverse_temperatures) == 1
                            else {"temps": inverse_temperatures})
        sampler = sampler_class(
            X_frame, y_series, sigma=sd_y, tau=tau, alpha=alpha_tree, beta=beta,
            iters=burnin + draws * thin, burnin=burnin, thinning=thin,
            store_tree_spacing=thin, max_stored_trees=draws,
            seed=np.random.default_rng(seed_sequences[chain]), light=True,
            **tempering_kwargs,
        )
        result = sampler.run()
        stored_trees = result["tree_store"]
        predictions = np.empty((draws, len(y)))
        for s, tree in enumerate(stored_trees):
            predictions[s], memberships = tree_draw_at_X(tree, X_frame)
            all_memberships.append(memberships)
        all_predictions.append(predictions)
        leaf_traces.append(result["tree_term_reg"])
        likelihood_traces.append(result["integr_llik_store"])
        mean_traces.append(predictions.mean(axis=1))
        counts = result["cache_counters"]
        chain_diagnostics.append({
            "chain": chain,
            "sampling_seconds": float(result["timings"]["elap_time"]),
            "seed_spawn_key": seed_sequences[chain].spawn_key,
            "acceptance_rate": counts["accepted"] / counts["proposed"],
            "mean_leaves": float(np.mean(result["tree_term_reg"])),
            "stump_fraction": float(np.mean(result["tree_term_reg"] == 1)),
        })
        if "PT_swap_stats" in result:
            swaps = result["PT_swap_stats"]
            chain_diagnostics[-1]["tempering_swap_rates"] = (
                swaps["accepts"] / np.maximum(swaps["attempts"], 1)).tolist()

    predictions = np.concatenate(all_predictions)
    selection_rng = np.random.default_rng(seed_sequences[-1])
    reporting_draw = int(selection_rng.integers(len(predictions)))
    reporting_leaves = all_memberships[reporting_draw]
    coverage, lengths, raw_lengths = [], [], []
    for indices in reporting_leaves:
        # Hold this reporting leaf fixed while averaging ALL posterior functions.
        leaf_draws = predictions[:, indices].mean(axis=1)
        lower, upper = np.quantile(leaf_draws, [level / 2, 1 - level / 2])
        target = mu[indices].mean()
        coverage.append(lower <= target <= upper)
        raw_lengths.append(float(upper - lower))
        lengths.append(float((upper - lower) * np.sqrt(len(indices))))

    rhats = {"leaf_count": _split_rhat(leaf_traces),
             "integrated_log_likelihood": _split_rhat(likelihood_traces),
             "grand_mean": _split_rhat(mean_traces)}
    if any(not np.isfinite(value) or value > 1.1 for value in rhats.values()):
        warnings.warn(f"BCART(beta={beta:g}, tau={tau:g}): inspect mixing; "
                      f"basic split R-hat = {rhats}", RuntimeWarning)
    diagnostics = {
        "elapsed_seconds": perf_counter() - fit_start,
        "chains": chain_diagnostics, "split_rhat": rhats,
        "seed": seed, "reporting_draw": reporting_draw,
        "reporting_leaf_sizes": [len(idx) for idx in reporting_leaves],
        "reporting_tree_MSE_to_mu": float(np.mean((predictions[reporting_draw] - mu)**2)),
        "raw_interval_lengths": raw_lengths,
    }
    return coverage, lengths, predictions.mean(axis=0), diagnostics


def segment_output_path(path, start, end):
    """Give every worker's [start, end) segment its own results file.

    Accept a common prefix ("bcart" or "bcart.pkl") or an already segmented
    filename ("bcart_0_50.pkl"). Calling the function directly with a shared
    prefix is safe across distinct segments, just as calling the CLI is.
    """
    path = Path(path)
    stem = path.stem if path.suffix == ".pkl" else path.name
    suffix = f"_{start}_{end}"
    if not stem.endswith(suffix):
        stem += suffix
    return path.with_name(stem + ".pkl")


def simulation_range(start=None, end=None, range_per_job=50, environ=None):
    """Use an explicit segment, a one-based SLURM task ID, or one local replicate."""
    if range_per_job < 1:
        raise ValueError("range_per_job must be positive")
    if (start is None) != (end is None):
        raise ValueError("Provide both --start and --end")
    if start is not None:
        if start < 0 or end <= start:
            raise ValueError("Require 0 <= start < end")
        return start, end
    environ = os.environ if environ is None else environ
    if "SLURM_ARRAY_TASK_ID" not in environ:
        return 0, 1
    try:
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
    except ValueError as exc:
        raise ValueError("SLURM_ARRAY_TASK_ID must be an integer") from exc
    if task_id < 1:
        raise ValueError("Automatic ranges require one-based SLURM task IDs; "
                         "use --start/--end for a different indexing convention")
    return (task_id - 1) * range_per_job, task_id * range_per_job


def terminal_inference_sim(n=200, p=5, a=1, b=2, sd_y=2,
                           noise_sd_list=(1.,), RRT_prop=.2,
                           start=0, end=1, level=.1, path=None, *,
                           beta_list=(1., 2.), tau=10., alpha_tree=.95,
                           burnin=5000, draws=2000, thin=5, chains=4,
                           bayes_seed=20261031,
                           inverse_temperatures=(1., .9, .75, .55, .35, .2, .08, .03)):
    """Copy toy_example's data/RRT pipeline; compare only RRT and BCART betas.

    Keep the existing BCART file's five-field results schema. SD(Y) is the
    fixed observation-noise SD, not the RRT randomization scale or BCART beta.
    """
    segment_start = perf_counter()
    if n < 50 or p < 3 or start < 0 or end <= start:
        raise ValueError("Require n >= 50, p >= 3, and 0 <= start < end")
    noise_sd_list, beta_list = tuple(noise_sd_list), tuple(beta_list)
    if not np.isfinite(sd_y) or sd_y <= 0:
        raise ValueError("sd_y must be positive and finite")
    if not np.all(np.isfinite([a, b])):
        raise ValueError("a and b must be finite")
    if not beta_list or any(not np.isfinite(b) or b < 0 for b in beta_list):
        raise ValueError("beta_list must contain finite nonnegative values")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    if not noise_sd_list or any(not np.isfinite(s) or s <= 0 for s in noise_sd_list):
        raise ValueError("noise_sd_list must contain positive finite values")
    if not 0 < alpha_tree < 1 or not 0 < level < 1 or bayes_seed < 0:
        raise ValueError("Require alpha_tree, level in (0, 1) and bayes_seed >= 0")
    if burnin < 1 or draws < 4 or thin < 1 or chains < 1:
        raise ValueError("Require burnin >= 1, draws >= 4, thin >= 1, chains >= 1")
    inverse_temperatures = _validate_temperatures(inverse_temperatures)
    if path is not None:
        path = segment_output_path(path, start, end)
        if not path.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {path.parent}")

    oper_char = {key: [] for key in ("Coverage Rate", "Length", "MSE", "Method", "SD(Y)")}
    diagnostics = {
        "setup": dict(experiment="toy_example", n=n, p=p, a=a, b=b, sd_y=sd_y,
                      noise_sd_list=noise_sd_list, RRT_prop=RRT_prop,
                      start=start, end=end, level=level, alpha_tree=alpha_tree,
                      beta_list=beta_list, tau=tau, mu_bar=0.,
                      max_depth=3, min_samples_split=50, min_bucket=20,
                      burnin=burnin, draws_per_chain=draws, thin=thin,
                      chains=chains, bayes_seed=bayes_seed,
                      inverse_temperatures=inverse_temperatures.tolist(),
                      tested_bayescart_commit="dd039db4b738bd981ec03dbdb96dda93b912fca4"),
        "versions": {name: version(name) for name in ("bayescart", "numpy", "scipy", "pandas")},
        "fits": [], "replications": [],
    }

    def record(method, coverage, lengths, pred, y_test, elapsed_seconds):
        oper_char["Coverage Rate"].append(np.mean(coverage))
        oper_char["Length"].append(np.mean(lengths))
        oper_char["MSE"].append(np.mean((y_test - pred)**2))
        oper_char["Method"].append(method)
        oper_char["SD(Y)"].append(sd_y)
        print(f"{method}, sigma={sd_y:g}: coverage={np.mean(coverage):.3f}, "
              f"length={np.mean(lengths):.3f}, elapsed={elapsed_seconds:.1f}s", flush=True)

    print(f"Worker segment [{start}, {end}): {path}; "
          f"n={n}, p={p}, sigma={sd_y:g}, a={a:g}, b={b:g}; "
          f"{len(noise_sd_list)} RRT + {len(beta_list)} BCART fits per replication",
          flush=True)

    for i in range(start, end):
        replication_start = perf_counter()
        print(i, "th simulation", flush=True)
        # Preserve toy_example.py's generation order and global RNG pipeline.
        np.random.seed(i + 1000)
        X = np.random.normal(size=(n, p))
        mu = b * ((X[:, 0] <= 0) * (
            1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
        y = mu + np.random.normal(size=(n,), scale=sd_y)
        y_test = generate_test(mu, sd_y)

        for noise_sd in noise_sd_list:
            method_start = perf_counter()
            reg_tree = RegressionTree(min_samples_split=50, max_depth=3,
                                      min_proportion=0., min_bucket=20)
            reg_tree.fit(X, y, sd=noise_sd * sd_y)
            fit_seconds = perf_counter() - method_start
            inference_start = perf_counter()
            coverage_i, lengths_i = randomized_inference(
                reg_tree=reg_tree, y=y, sd_y=sd_y, mu=mu, level=level,
                noise_sd=noise_sd, prop=RRT_prop)
            inference_seconds = perf_counter() - inference_start
            pred_test = reg_tree.predict(X)
            elapsed_seconds = perf_counter() - method_start
            method = "RRT" if noise_sd_list == (1.,) else f"RRT({noise_sd:g})"
            record(method, coverage_i, lengths_i, pred_test, y_test, elapsed_seconds)
            diagnostics["fits"].append({
                "replication": i, "SD(Y)": sd_y, "Method": method,
                "noise_sd": noise_sd, "elapsed_seconds": elapsed_seconds,
                "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
                "reporting_leaf_sizes": [int(node.membership.sum())
                                         for node in reg_tree.terminal_nodes],
            })

        for beta in beta_list:
            # Separate generator: BCART never changes RRT's global RNG state.
            # Encode float bits so seeds are stable when the beta grid is reordered.
            bits = np.asarray([sd_y, beta, tau], dtype="<f8").view("<u4").tolist()
            seed = [bayes_seed, i, *bits]
            method = f"BCART(beta={beta:g}, tau={tau:g})"
            coverage, lengths, pred, fit_diag = bayesian_cart_inference(
                X, y, mu, sd_y, beta=beta, tau=tau, alpha_tree=alpha_tree,
                level=level, burnin=burnin, draws=draws, thin=thin,
                chains=chains, seed=seed, inverse_temperatures=inverse_temperatures)
            record(method, coverage, lengths, pred, y_test, fit_diag["elapsed_seconds"])
            diagnostics["fits"].append(dict(
                replication=i, Method=method, **{"SD(Y)": sd_y}, **fit_diag))

        elapsed_seconds = perf_counter() - replication_start
        diagnostics["replications"].append({"replication": i, "elapsed_seconds": elapsed_seconds})
        diagnostics["elapsed_seconds"] = perf_counter() - segment_start
        if path is not None:
            joblib.dump(oper_char, path)
            joblib.dump(diagnostics, str(Path(path).with_suffix(".diagnostics.pkl")))
        print(f"Replication {i} completed in {elapsed_seconds / 60:.2f} minutes; "
              f"checkpoint={path}", flush=True)
    return oper_char


def _float_list(value):
    try:
        values = [float(x) for x in value.strip("[]").split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("The list must not be empty")
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", help="Output prefix; writes prefix_start_end.pkl")
    parser.add_argument("--start", type=int, help="First replication (inclusive); requires --end")
    parser.add_argument("--end", type=int, help="Last replication (exclusive); requires --start")
    parser.add_argument("--range-per-job", type=int, default=50,
                        help="Replications per one-based SLURM array task; default 50")
    parser.add_argument("--betas", type=_float_list, default=[1., 2.],
                        help="BCART tree-prior betas; default [1,2]")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--a", type=float, default=1.)
    parser.add_argument("--b", type=float, default=2.)
    parser.add_argument("--sd-y", type=float, default=2., help="Fixed observation-noise SD")
    parser.add_argument("--tau", type=float, default=10., help="Fixed BCART leaf-mean prior SD")
    parser.add_argument("--alpha-tree", type=float, default=.95)
    parser.add_argument("--level", type=float, default=.1, help="Miscoverage probability")
    parser.add_argument("--rrt-noise-sds", type=_float_list, default=[1.],
                        help="RRT randomization SD / sigma; default [1]")
    parser.add_argument("--rrt-prop", type=float, default=.2)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--draws", type=int, default=2000, help="Retained posterior draws per chain")
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--inverse-temperatures", type=_float_list,
                        default=[1., .9, .75, .55, .35, .2, .08, .03],
                        help="One value [1] or a decreasing even-length tempering ladder")
    parser.add_argument("--bayes-seed", type=int, default=20261031)
    args = parser.parse_args()
    try:
        start, end = simulation_range(args.start, args.end, args.range_per_job)
    except ValueError as exc:
        parser.error(str(exc))
    terminal_inference_sim(
        n=args.n, p=args.p, a=args.a, b=args.b, sd_y=args.sd_y,
        noise_sd_list=args.rrt_noise_sds, RRT_prop=args.rrt_prop,
        start=start, end=end, level=args.level,
        path=args.prefix,
        beta_list=args.betas, tau=args.tau, alpha_tree=args.alpha_tree,
        burnin=args.burnin, draws=args.draws, thin=args.thin,
        chains=args.chains, bayes_seed=args.bayes_seed,
        inverse_temperatures=args.inverse_temperatures,
    )
