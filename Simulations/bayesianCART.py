r"""Gaussian vary-signal comparison: RRT(1) versus single-tree Bayesian CART.

Requires Python >= 3.12 (BayesCART's requirement). Install in a separate env:
    python -m pip install \
        'bayescart @ git+https://github.com/guglielmogattiglio/bayescart.git@dd039db4b738bd981ec03dbdb96dda93b912fca4' \
        cvxpy

Run with the same four positional arguments as vary_signal.py:
    python Simulations/bayesianCART.py 0 500 '[1,2,5,10]' results/bcart

This writes results/bcart_0_500.pkl, a dictionary of parallel lists with keys
"Coverage Rate", "Length", "MSE", "Method", "SD(Y)". Length is CI width times
sqrt(leaf size), exactly as in vary_signal.py. A separate .diagnostics.pkl stores
configuration, chain diagnostics, reporting-tree sizes, and package versions.
The output directory must exist. Files are checkpointed after each replication.

Defaults: alpha_tree=.95; beta_tree in (1, 2); independent leaf means N(0, tau^2)
with tau in (5, 10, 20); known common observation SD. Tree depth <= 3,
min_samples_split=50, min_bucket=20. BayesCART supplies the tree representation,
grow/prune/change proposals, MCMC loop, and storage. The adapter below replaces
its unknown leaf-specific variance model and adjusts the MH ratio for our prior
and tree restrictions. Swap moves are disabled. No R or Tree-Values dependency.

For each fit, a uniformly selected retained posterior tree defines the reporting
partition. Every retained draw of the latent regression function is averaged
over each fixed reporting leaf; the 5% and 95% quantiles give the default 90% CI.
MSE uses the full-posterior mean at the training X and independent test responses,
matching the fixed-design test experiment in vary_signal.py.

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
import itertools
from pathlib import Path
import sys
import warnings
from importlib.metadata import version

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
        "chains": chain_diagnostics, "split_rhat": rhats,
        "seed": seed, "reporting_draw": reporting_draw,
        "reporting_leaf_sizes": [len(idx) for idx in reporting_leaves],
        "reporting_tree_MSE_to_mu": float(np.mean((predictions[reporting_draw] - mu)**2)),
        "raw_interval_lengths": raw_lengths,
    }
    return coverage, lengths, predictions.mean(axis=0), diagnostics


def vary_signal_sim(n=200, p=10, sd_y_list=(1, 2, 5, 10), noise_sd=1,
                    start=0, end=100, level=.1, path=None, *,
                    beta_list=(1., 2.), tau_list=(5., 10., 20.), alpha_tree=.95,
                    burnin=5000, draws=2000, thin=5, chains=4,
                    bayes_seed=20261031,
                    inverse_temperatures=(1., .9, .75, .55, .35, .2, .08, .03)):
    """Same data, RRT, and result schema as vary_signal.py; replace other baselines."""
    if n < 50 or p < 3 or start < 0 or end <= start:
        raise ValueError("Require n >= 50, p >= 3, and 0 <= start < end")
    sd_y_list, beta_list, tau_list = tuple(sd_y_list), tuple(beta_list), tuple(tau_list)
    if not sd_y_list or any(not np.isfinite(s) or s <= 0 for s in sd_y_list):
        raise ValueError("sd_y_list must contain positive finite values")
    if not beta_list or any(not np.isfinite(b) or b < 0 for b in beta_list):
        raise ValueError("beta_list must contain finite nonnegative values")
    if not tau_list or any(not np.isfinite(t) or t <= 0 for t in tau_list):
        raise ValueError("tau_list must contain positive finite values")
    if not 0 < alpha_tree < 1 or not 0 < level < 1 or bayes_seed < 0:
        raise ValueError("Require alpha_tree, level in (0, 1) and bayes_seed >= 0")
    if burnin < 1 or draws < 4 or thin < 1 or chains < 1:
        raise ValueError("Require burnin >= 1, draws >= 4, thin >= 1, chains >= 1")
    inverse_temperatures = _validate_temperatures(inverse_temperatures)
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("noise_sd must be positive and finite")
    if path is not None and not Path(path).parent.is_dir():
        raise ValueError(f"Output directory does not exist: {Path(path).parent}")

    oper_char = {key: [] for key in ("Coverage Rate", "Length", "MSE", "Method", "SD(Y)")}
    diagnostics = {
        "setup": dict(n=n, p=p, sd_y_list=sd_y_list, noise_sd=noise_sd,
                      start=start, end=end, level=level, alpha_tree=alpha_tree,
                      beta_list=beta_list, tau_list=tau_list, mu_bar=0.,
                      max_depth=3, min_samples_split=50, min_bucket=20,
                      burnin=burnin, draws_per_chain=draws, thin=thin,
                      chains=chains, bayes_seed=bayes_seed,
                      inverse_temperatures=inverse_temperatures.tolist(),
                      tested_bayescart_commit="dd039db4b738bd981ec03dbdb96dda93b912fca4"),
        "versions": {name: version(name) for name in ("bayescart", "numpy", "scipy", "pandas")},
        "fits": [],
    }

    def record(method, sd_y, coverage, lengths, pred, y_test):
        oper_char["Coverage Rate"].append(np.mean(coverage))
        oper_char["Length"].append(np.mean(lengths))
        oper_char["MSE"].append(np.mean((y_test - pred)**2))
        oper_char["Method"].append(method)
        oper_char["SD(Y)"].append(sd_y)
        print(f"{method}, sigma={sd_y:g}: coverage={np.mean(coverage):.3f}, "
              f"length={np.mean(lengths):.3f}", flush=True)

    for i in range(start, end):
        for sd_y in sd_y_list:
            print(i, "th simulation", flush=True)
            # Preserve legacy RNG and draw order through RRT fit and inference.
            np.random.seed(i + 1000)
            X = np.random.normal(size=(n, p))
            a, b = 1, 2
            mu = b * ((X[:, 0] <= 0) * (
                1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
            y = mu + np.random.normal(size=(n,), scale=sd_y)
            y_test = generate_test(mu, sd_y)
            reg_tree = RegressionTree(min_samples_split=50, max_depth=3,
                                      min_proportion=0., min_bucket=20)
            reg_tree.fit(X, y, sd=noise_sd * sd_y)
            coverage_i, lengths_i = randomized_inference(
                reg_tree=reg_tree, y=y, sd_y=sd_y, mu=mu, level=level,
                noise_sd=noise_sd, reduced_dim=None, prop=.2)
            record("RRT", sd_y, coverage_i, lengths_i, reg_tree.predict(X), y_test)
            diagnostics["fits"].append({
                "replication": i, "SD(Y)": sd_y, "Method": "RRT",
                "reporting_leaf_sizes": [int(node.membership.sum())
                                         for node in reg_tree.terminal_nodes],
            })

            for beta, tau in itertools.product(beta_list, tau_list):
                # Separate generator: BCART never changes RRT's global RNG state.
                # Encode float bits so seeds are stable when the parameter grid is reordered.
                bits = np.asarray([sd_y, beta, tau], dtype="<f8").view("<u4").tolist()
                seed = [bayes_seed, i, *bits]
                method = f"BCART(beta={beta:g}, tau={tau:g})"
                coverage, lengths, pred, fit_diag = bayesian_cart_inference(
                    X, y, mu, sd_y, beta=beta, tau=tau, alpha_tree=alpha_tree,
                    level=level, burnin=burnin, draws=draws, thin=thin,
                    chains=chains, seed=seed, inverse_temperatures=inverse_temperatures)
                record(method, sd_y, coverage, lengths, pred, y_test)
                diagnostics["fits"].append(dict(
                    replication=i, Method=method, **{"SD(Y)": sd_y}, **fit_diag))

        if path is not None:
            joblib.dump(oper_char, path)
            joblib.dump(diagnostics, str(Path(path).with_suffix(".diagnostics.pkl")))
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
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("sd_y_list", type=_float_list)
    parser.add_argument("prefix", help="Output prefix; writes prefix_start_end.pkl")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--betas", type=_float_list, default=[1., 2.])
    parser.add_argument("--taus", type=_float_list, default=[5., 10., 20.])
    parser.add_argument("--alpha-tree", type=float, default=.95)
    parser.add_argument("--level", type=float, default=.1, help="Miscoverage probability")
    parser.add_argument("--noise-sd", type=float, default=1., help="RRT randomization SD / sigma")
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--draws", type=int, default=2000, help="Retained posterior draws per chain")
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--inverse-temperatures", type=_float_list,
                        default=[1., .9, .75, .55, .35, .2, .08, .03],
                        help="One value [1] or a decreasing even-length tempering ladder")
    parser.add_argument("--bayes-seed", type=int, default=20261031)
    args = parser.parse_args()
    vary_signal_sim(
        n=args.n, p=args.p, sd_y_list=args.sd_y_list, noise_sd=args.noise_sd,
        start=args.start, end=args.end, level=args.level,
        path=f"{args.prefix}_{args.start}_{args.end}.pkl",
        beta_list=args.betas, tau_list=args.taus, alpha_tree=args.alpha_tree,
        burnin=args.burnin, draws=args.draws, thin=args.thin,
        chains=args.chains, bayes_seed=args.bayes_seed,
        inverse_temperatures=args.inverse_temperatures,
    )
