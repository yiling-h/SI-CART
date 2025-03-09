import sys, os

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

# For tree-values
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

# Select a CRAN mirror to download from
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)  # Select the first mirror

# Install 'remotes' if it's not already installed
if not rpackages.isinstalled('remotes'):
    utils.install_packages(StrVector(('remotes',)))

import rpy2.robjects as ro

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

def tree_values_inference(X, y, mu, sd_y, max_depth=5, level=0.1,
                          X_test=None):
    # Convert the NumPy matrix to an R matrix
    X_r = numpy2ri.py2rpy(X)
    y_r = numpy2ri.py2rpy(y)

    # Assign the R matrix to a variable in the R environment (optional)
    ro.globalenv['X_r'] = X_r
    ro.globalenv['y_r'] = y_r
    ro.globalenv['p'] = X.shape[1]

    # Construct dataset
    ro.r('data <- cbind(y_r, X_r)')
    # Set the column names to "y", "x1", "x2", ..., "x10"
    ro.r('colnames(data) <- c("y", paste0("x", 1:p))')
    ro.r('data = data.frame(data)')

    # Define the rpart tree model
    tree_cmd = ('bls.tree <- rpart(y ~ ., data=data, model = TRUE, ' +
                'control = rpart.control(cp=0.00, minsplit = 50, minbucket = 20, maxdepth=') + str(max_depth) + '))'
    ro.r(tree_cmd)
    bls_tree = ro.r('bls.tree')
    # Plot the tree values (this will plot directly if you have a plotting backend set up)
    # ro.r('treeval.plot(bls.tree, inferenceType=0)')

    # ro.r('print(row.names(bls.tree$frame)[bls.tree$frame$var == "<leaf>"])')
    ro.r('leaf_idx <- (row.names(bls.tree$frame)[bls.tree$frame$var == "<leaf>"])')
    leaf_idx = ro.r['leaf_idx']

    # Get node mapping
    ro.r('idx_full <- 1:nrow(bls.tree$frame)')
    ro.r('mapped_idx <- idx_full[bls.tree$frame$var == "<leaf>"]')

    len = []
    coverage = []
    len_naive = []
    coverage_naive = []

    for i, idx in enumerate(leaf_idx):
        # Get the branch information for a specific branch in the tree
        command = 'branch <- getBranch(bls.tree, ' + str(idx) + ')'
        ro.r(command)
        # Perform branch inference
        ro.r(f'result <- branchInference(bls.tree, branch, type="reg", alpha = 0.10, sigma_y={sd_y})')
        # Get confidence intervals
        confint = ro.r('result$confint')
        len.append(confint[1] - confint[0])

        target_cmd = "contrast <- (bls.tree$where == mapped_idx[" + str(i + 1) + "])"
        ro.r(target_cmd)
        contrast = ro.r('contrast')
        contrast = np.array(contrast)

        contrast = np.array(contrast * 1 / np.sum(contrast))

        target = contrast.dot(mu)
        coverage.append(target >= confint[0] and target <= confint[1])

        # Naive after tree value
        # Confidence intervals
        naive_CI = [contrast.dot(y) -
                    np.linalg.norm(contrast) * sd_y * ndist.ppf(1 - level / 2),
                    contrast.dot(y) +
                    np.linalg.norm(contrast) * sd_y * ndist.ppf(1 - level / 2)]
        coverage_naive.append((target >= naive_CI[0] and target <= naive_CI[1]))
        len_naive.append(naive_CI[1] - naive_CI[0])

    if X_test is not None:
        X_test_r = numpy2ri.py2rpy(X_test)
        ro.globalenv['X_test_r'] = X_test_r
        ro.r('pred <- predict(bls.tree, data = X_test_r)')
        pred = ro.r['pred']
    else:
        pred = None

    return (np.mean(coverage), np.mean(len),
            np.mean(coverage_naive), np.mean(len_naive), pred)

def generate_test(mu, sd_y):
    n = mu.shape[0]
    return mu + np.random.normal(size=(n,), scale=sd_y)


def randomized_inference(reg_tree, sd_y, y, mu, level=0.1):
    # print(reg_tree.terminal_nodes)
    coverage_i = []
    lengths_i = []

    for node in reg_tree.terminal_nodes:
        """pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs, marginal \
            = (reg_tree.condl_node_inference(node=node,
                                             ngrid=10000,
                                             ncoarse=100,
                                             grid_w_const=1.5,
                                             reduced_dim=None,
                                             sd=sd_y,
                                             use_cvxpy=True))"""
        pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs, _ \
            = (reg_tree.node_inference(node=node,
                                       ngrid=10000,
                                       ncoarse=50,
                                       grid_w_const=30,
                                       sd=sd_y,
                                       use_cvxpy=True,
                                       query_grid=True,
                                       query_size=50))

        target = contrast.dot(mu)

        # This is an interval for
        # eta_*'mu = eta'mu / (norm(eta) * sd_y)
        selective_CI = (dist.equal_tailed_interval(observed=norm_contrast.dot(y),
                                                   alpha=level))
        selective_CI = np.array(selective_CI)
        selective_CI *= np.linalg.norm(contrast) * sd_y
        coverage_i.append((target >= selective_CI[0] and target <= selective_CI[1]))
        lengths_i.append(selective_CI[1] - selective_CI[0])

    return coverage_i, lengths_i


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
        lengths.append(CI[1] - CI[0])

    if X_test is not None:
        pred = reg_tree.predict(X_test)
    else:
        pred = None

    return coverage, lengths, pred
# %%

def terminal_inference_sim(n=50, p=5, a=0.1, b=0.1,
                           sd_y=1,
                           noise_sd_list=[0.5, 1, 2, 5],
                           UV_gamma_list=[],
                           use_nonrand=True,
                           start=0, end=100,
                           level=0.1, path=None):
    method_list = noise_sd_list.copy()
    if use_nonrand:
        method_list += ["Tree val", "Naive"]
    for gamma in UV_gamma_list:
        method_list.append("UV_" + str(gamma))

    coverage_dict = {m: [] for m in method_list}
    length_dict = {m: [] for m in method_list}
    MSE_dict = {m: [] for m in method_list}

    for i in range(start, end):
        print(i, "th simulation")
        np.random.seed(i + 1000)
        X = np.random.normal(size=(n, p))

        mu = b * ((X[:, 0] <= 0) * (1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
        y = mu + np.random.normal(size=(n,), scale=sd_y)
        y_test = generate_test(mu, sd_y)

        for noise_sd in noise_sd_list:
            # Create and train the regression tree
            reg_tree = RegressionTree(min_samples_split=50, max_depth=3,
                                      min_proportion=0., min_bucket=20)

            reg_tree.fit(X, y, sd=noise_sd * sd_y)

            coverage_i, lengths_i = randomized_inference(reg_tree=reg_tree,
                                                         y=y, sd_y=sd_y, mu=mu,
                                                         level=level)
            pred_test = reg_tree.predict(X)
            MSE_test = (np.mean((y_test - pred_test) ** 2))
            # Record results
            coverage_dict[noise_sd].append(np.mean(coverage_i))
            length_dict[noise_sd].append(np.mean(lengths_i))
            MSE_dict[noise_sd].append(MSE_test)

        if use_nonrand:
            # Tree value & naive inference & prediction
            (coverage_treeval, avg_len_treeval,
             coverage_treeval_naive, avg_len_treeval_naive,
             pred_test_treeval) = tree_values_inference(X, y, mu, sd_y=sd_y,
                                                        X_test=X, max_depth=3)
            MSE_test_treeval = (np.mean((y_test - pred_test_treeval) ** 2))

            coverage_dict["Tree val"].append(coverage_treeval)
            length_dict["Tree val"].append(avg_len_treeval)
            MSE_dict["Tree val"].append(MSE_test_treeval)
            coverage_dict["Naive"].append(coverage_treeval_naive)
            length_dict["Naive"].append(avg_len_treeval_naive)
            MSE_dict["Naive"].append(MSE_test_treeval)

        for gamma in UV_gamma_list:
            gamma_key = "UV_" + str(gamma)
            # UV decomposition
            coverage_UV, len_UV, pred_UV = UV_decomposition(X, y, mu, sd_y, X_test=X,
                                                            min_prop=0., max_depth=3,
                                                            min_sample=50, min_bucket=20,
                                                            gamma=gamma)
            MSE_UV = (np.mean((y_test - pred_UV) ** 2))
            coverage_dict[gamma_key].append(np.mean(coverage_UV))
            length_dict[gamma_key].append(np.mean(len_UV))
            MSE_dict[gamma_key].append(MSE_UV)

        if path is not None:
            joblib.dump([coverage_dict, length_dict, MSE_dict], path, compress=1)

    return coverage_dict, length_dict, MSE_dict


if __name__ == '__main__':

    argv = sys.argv
    ## sys.argv: [something, start, end, n, p, y_sd, omega_sd]
    start, end= (int(argv[1]), int(argv[2]))
    # Parse the first list from the third argument
    noise_sd_list = [float(x) for x in sys.argv[3].strip("[]").split(",") if x]
    # Parse the second list from the fourth argument
    UV_gamma_list = [float(x) for x in sys.argv[4].strip("[]").split(",") if x]
    use_nonrand = bool(argv[5])
    prefix = argv[6]

    # Activate automatic conversion between pandas and R data frames
    pandas2ri.activate()

    # Import R libraries
    treevalues = importr('treevalues')
    rpart = importr('rpart')

    # start, end, randomizer_scale, ncores = 0, 40, 1.5, 4
    dir = (prefix + '_' + str(start) + '_' + str(end) + '.pkl')

    (coverage_dict, length_dict, MSE_dict) \
        = terminal_inference_sim(start=start, end=end, n=200, p=5, sd_y=2,
                                 noise_sd_list=noise_sd_list,
                                 UV_gamma_list=UV_gamma_list,
                                 use_nonrand=use_nonrand,
                                 a=1,b=2, level=0.1, path=dir)

    joblib.dump([coverage_dict, length_dict, MSE_dict], dir, compress=1)