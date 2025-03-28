import sys, os

import numpy as np

# Get the path to the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the path to the outermost directory (project directory in this case)
# The '..' means go up one directory from 'subdirectory' to the outermost directory
module_path = os.path.abspath(os.path.join(current_dir, '..'))

# Add this path to sys.path if it's not already there
if module_path not in sys.path:
    sys.path.append(module_path)


from CART import RegressionTree
from Utils.plotting import  *
from scipy.stats import norm as ndist
import joblib
def generate_test(mu, sd_y):
    n = mu.shape[0]
    return mu + np.random.normal(size=(n,), scale=sd_y)

def UV_decomposition(X, y, mu, sd_y,
                     max_depth=5, min_prop=0, min_sample=10, min_bucket=5,
                     level=0.1, gamma=1,
                     X_test=None):
    n = X.shape[0]
    W = np.random.normal(loc=0, scale=sd_y * np.sqrt(gamma), size=(n,))
    U = y + W
    V = y - W / gamma
    sd_V = sd_y * np.sqrt(1 + 1 / gamma)
    reg_tree = RegressionTree(min_samples_split=min_sample, max_depth=max_depth,
                              min_proportion=min_prop, min_bucket=min_bucket)
    reg_tree.fit(X, U, sd=0)

    coverage = []
    lengths = []

    for node in reg_tree.terminal_nodes:
        contrast = node.membership

        contrast = np.array(contrast * 1 / np.sum(contrast))

        target = contrast.dot(mu)

        # Naive after tree value
        # Confidence intervals
        CI = [contrast.dot(V) -
              np.linalg.norm(contrast) * sd_V * ndist.ppf(1 - level / 2),
              contrast.dot(V) +
              np.linalg.norm(contrast) * sd_V * ndist.ppf(1 - level / 2)]
        coverage.append((target >= CI[0] and target <= CI[1]))
        root_n = 1/np.linalg.norm(contrast)
        lengths.append((CI[1] - CI[0]) * root_n)

    if X_test is not None:
        pred = reg_tree.predict(X_test)
    else:
        pred = None

    return coverage, lengths, pred

def randomized_inference(reg_tree, sd_y, y, mu, noise_sd=1,
                         level=0.1, reduced_dim=5, prop=0.05):
    # print(reg_tree.terminal_nodes)
    coverage_i = []
    lengths_i = []

    for node in reg_tree.terminal_nodes:
        if prop != 1.0:
            (pval, dist, contrast, norm_contrast, obs_tar, logW, suff,
             sel_probs, ref_hat_layer, marginal) \
                = (reg_tree.condl_node_inference(node=node,
                                                 ngrid=10000,
                                                 ncoarse=100,
                                                 grid_w_const=5*noise_sd,
                                                 query_size=100,
                                                 query_grid=False,
                                                 reduced_dim=reduced_dim,
                                                 prop=prop,
                                                 sd=sd_y,
                                                 use_cvxpy=False))
        else:
            pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs, _ \
                = (reg_tree.node_inference(node=node,
                                           ngrid=10000,
                                           ncoarse=100,
                                           grid_w_const=5*noise_sd,
                                           sd=sd_y,
                                           use_cvxpy=False,
                                           query_grid=False,
                                           query_size=100, interp_kind='quadratic'))

        target = contrast.dot(mu)

        # This is an interval for
        # eta_*'mu = eta'mu / (norm(eta) * sd_y)
        selective_CI = (dist.equal_tailed_interval(observed=norm_contrast.dot(y),
                                                   alpha=level))
        selective_CI = np.array(selective_CI)
        selective_CI *= sd_y #np.linalg.norm(contrast) * sd_y
        coverage_i.append((target >= selective_CI[0] and target <= selective_CI[1]))
        lengths_i.append(selective_CI[1] - selective_CI[0])

    return coverage_i, lengths_i
# %%

def terminal_inference_sim(n=50, p=5, a=0.1, b=0.1,
                           sd_y=1,
                           r_list=[0.5, 1, 2, 5],
                           noise_sd=1,
                           start=0, end=100,
                           level=0.1, path=None):
    num_r = len(r_list)
    r_list = r_list.copy()
    r_list.append('UV(0.1)')
    r_list.append('full')

    coverage_dict = {m: [] for m in r_list}
    length_dict = {m: [] for m in r_list}
    MSE_dict = {m: [] for m in r_list}

    for i in range(start, end):
        print(i, "th simulation")
        np.random.seed(i + 1000)
        X = np.random.normal(size=(n, p))

        mu = b * ((X[:, 0] <= 0) * (1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
        y = mu + np.random.normal(size=(n,), scale=sd_y)
        y_test = generate_test(mu, sd_y)

        for r in r_list[0:num_r]:
            if r < 1:
                reduced_dim = None
                prop = r
            else:
                reduced_dim = r
                prop = None

            # Create and train the regression tree
            reg_tree = RegressionTree(min_samples_split=10, max_depth=2,
                                      min_proportion=0., min_bucket=3)

            reg_tree.fit(X, y, sd=noise_sd * sd_y)

            coverage_i, lengths_i = randomized_inference(reg_tree=reg_tree,
                                                         y=y, sd_y=sd_y, mu=mu, noise_sd=noise_sd,
                                                         level=level, reduced_dim=reduced_dim, prop=prop)
            pred_test = reg_tree.predict(X)
            MSE_test = (np.mean((y_test - pred_test) ** 2))
            # Record results
            coverage_dict[r].append(np.mean(coverage_i))
            length_dict[r].append(np.mean(lengths_i))
            MSE_dict[r].append(MSE_test)

        coverage_UV, len_UV, pred_UV = UV_decomposition(X, y, mu, sd_y, X_test=X,
                                                        min_prop=0., max_depth=2,
                                                        min_sample=10, min_bucket=3,
                                                        gamma=0.1)
        MSE_UV = (np.mean((y_test - pred_UV) ** 2))
        coverage_dict['UV(0.1)'].append(np.mean(coverage_UV))
        length_dict['UV(0.1)'].append(np.mean(len_UV))
        MSE_dict['UV(0.1)'].append(MSE_UV)

        coverage_full, lengths_full = randomized_inference(reg_tree=reg_tree,
                                                           y=y, sd_y=sd_y, mu=mu, noise_sd=noise_sd,
                                                           level=level, reduced_dim=reduced_dim, prop=1.0)
        pred_full = reg_tree.predict(X)
        MSE_full = (np.mean((y_test - pred_full) ** 2))
        # Record results
        coverage_dict['full'].append(np.mean(coverage_full))
        length_dict['full'].append(np.mean(lengths_full))
        MSE_dict['full'].append(MSE_full)


        if path is not None:
            joblib.dump([coverage_dict, length_dict, MSE_dict], path, compress=1)

    return coverage_dict, length_dict, MSE_dict


if __name__ == '__main__':

    argv = sys.argv
    ## sys.argv: [something, start, end, omega_sd, prefix]
    start, end= (int(argv[1]), int(argv[2]))
    noise_sd = float(argv[3])
    prefix = argv[4]

    # start, end, randomizer_scale, ncores = 0, 40, 1.5, 4
    dir = (f"{prefix}_noisesd_{noise_sd}_{start}_{end}.pkl")

    (coverage_dict, length_dict, MSE_dict) \
        = terminal_inference_sim(start=start, end=end, n=50, p=5, sd_y=2, noise_sd=noise_sd,
                                 r_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                 a=1,b=2, level=0.1, path=dir)

    joblib.dump([coverage_dict, length_dict, MSE_dict], dir, compress=1)