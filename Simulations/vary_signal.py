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


def tree_values_inference(X, y, mu, max_depth=5, level=0.1,
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
                'control = rpart.control(cp=0.00, minsplit = 25, minbucket = 10, maxdepth=') + str(max_depth) + '))'
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

    for i, idx in enumerate(leaf_idx):
        # Get the branch information for a specific branch in the tree
        command = 'branch <- getBranch(bls.tree, ' + str(idx) + ')'
        ro.r(command)
        # Perform branch inference
        ro.r('result <- branchInference(bls.tree, branch, type="reg", alpha = 0.10)')
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

    if X_test is not None:
        X_test_r = numpy2ri.py2rpy(X_test)
        ro.globalenv['X_test_r'] = X_test_r
        ro.r('pred <- predict(bls.tree, data = X_test_r)')
        pred = ro.r['pred']
    else:
        pred = None

    return (np.mean(coverage), np.mean(len), pred)

def generate_test(mu, sd_y):
    n = mu.shape[0]
    return mu + np.random.normal(size=(n,), scale=sd_y)


def randomized_inference(reg_tree, sd_y, y, mu, level=0.1):
    # print(reg_tree.terminal_nodes)
    coverage_i = []
    lengths_i = []

    for node in reg_tree.terminal_nodes:
        pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs \
            = (reg_tree.condl_node_inference(node=node,
                                             ngrid=10000,
                                             ncoarse=50,
                                             grid_w_const=3,
                                             reduced_dim=1,
                                             sd=sd_y,
                                             use_cvxpy=True))
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
# %%

import itertools


def vary_signal_sim(n=50, p=5, sd_y_list=[1, 2, 5, 10], noise_sd=1,
                    start=0, end=100, level=0.1, path=None):
    oper_char = {}
    oper_char["Coverage Rate"] = []
    oper_char["Length"] = []
    oper_char["MSE"] = []
    oper_char["Method"] = []
    oper_char["SD(Y)"] = []
    # oper_char["a"] = []
    # oper_char["b"] = []
    a = 1
    b = 2

    # for ab_prod in itertools.product(a_list, b_list):
    # a = ab_prod[0]
    # b = ab_prod[1]
    for i in range(start, end):
        for sd_y in sd_y_list:
            print(i, "th simulation")
            # np.random.seed(i + 48105)
            X = np.random.normal(size=(n, p))

            mu = b * ((X[:, 0] <= 0) * (1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
            y = mu + np.random.normal(size=(n,), scale=sd_y)
            y_test = generate_test(mu, sd_y)
            hat_sd_y = np.std(y)

            # Create and train the regression tree
            reg_tree = RegressionTree(min_samples_split=50, max_depth=3,
                                      min_proportion=0., min_bucket=20)
            reg_tree.fit(X, y, sd=noise_sd * hat_sd_y)

            # RRT Inference
            coverage_i, lengths_i = randomized_inference(reg_tree=reg_tree,
                                                         y=y, sd_y=hat_sd_y, mu=mu,
                                                         level=level)
            pred_test = reg_tree.predict(X)
            MSE_test = (np.mean((y_test - pred_test) ** 2))
            # Record results
            oper_char["Coverage Rate"].append(np.mean(coverage_i))
            oper_char["Length"].append(np.mean(lengths_i))
            oper_char["MSE"].append(MSE_test)
            oper_char["Method"].append("RRT")
            oper_char["SD(Y)"].append(sd_y)
            # oper_char["a"].append(a)
            # oper_char["b"].append(b)

            # Tree value & naive inference & prediction
            (coverage_treeval, avg_len_treeval,
             pred_test_treeval) = tree_values_inference(X, y, mu,
                                                        X_test=X, max_depth=3)
            MSE_test_treeval = (np.mean((y_test - pred_test_treeval) ** 2))

            oper_char["Coverage Rate"].append(coverage_treeval)
            oper_char["Length"].append(avg_len_treeval)
            oper_char["MSE"].append(MSE_test_treeval)
            oper_char["Method"].append("Tree-Values")
            oper_char["SD(Y)"].append(sd_y)
            # oper_char["a"].append(a)
            # oper_char["b"].append(b)

        if path is not None:
            joblib.dump(oper_char, path)

    return oper_char


if __name__ == '__main__':

    argv = sys.argv
    ## sys.argv: [something, start, end, sd_y_list, prefix]
    start, end= (int(argv[1]), int(argv[2]))
    # Parse the first list from the third argument
    sd_y_list = [float(x) for x in sys.argv[3].strip("[]").split(",") if x]
    prefix = argv[4]

    # Activate automatic conversion between pandas and R data frames
    pandas2ri.activate()

    # Import R libraries
    treevalues = importr('treevalues')
    rpart = importr('rpart')

    # start, end, randomizer_scale, ncores = 0, 40, 1.5, 4
    dir = (prefix + '_' + str(start) + '_' + str(end) + '.pkl')

    oper_char = vary_signal_sim(n=200, p=10, sd_y_list=sd_y_list,
                                noise_sd=1,
                                start=start, end=end, level=0.1, path=dir)

    joblib.dump(oper_char, dir)