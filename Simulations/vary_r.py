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
def generate_test(mu, sd_y):
    n = mu.shape[0]
    return mu + np.random.normal(size=(n,), scale=sd_y)

def randomized_inference(reg_tree, sd_y, y, mu,
                         level=0.1, reduced_dim=5, prop=0.05):
    # print(reg_tree.terminal_nodes)
    coverage_i = []
    lengths_i = []

    for node in reg_tree.terminal_nodes:
        (pval, dist, contrast, norm_contrast, obs_tar, logW, suff,
         sel_probs, ref_hat_layer, marginal) \
            = (reg_tree.condl_node_inference(node=node,
                                             ngrid=10000,
                                             ncoarse=300,
                                             grid_w_const=5,
                                             query_size=100,
                                             query_grid=False,
                                             reduced_dim=reduced_dim,
                                             prop=prop,
                                             sd=sd_y,
                                             use_cvxpy=False))
        """pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs, _ \
            = (reg_tree.node_inference(node=node,
                                       ngrid=10000,
                                       ncoarse=50,
                                       grid_w_const=30,
                                       sd=sd_y,
                                       use_cvxpy=False,
                                       query_grid=True,
                                       query_size=200, interp_kind='linear'))"""

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

def terminal_inference_sim(n=50, p=5, a=0.1, b=0.1,
                           sd_y=1,
                           r_list=[0.5, 1, 2, 5],
                           noise_sd=1,
                           start=0, end=100,
                           level=0.1, path=None):

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

        for r in r_list:
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
                                                         y=y, sd_y=sd_y, mu=mu,
                                                         level=level, reduced_dim=reduced_dim, prop=prop)
            pred_test = reg_tree.predict(X)
            MSE_test = (np.mean((y_test - pred_test) ** 2))
            # Record results
            coverage_dict[r].append(np.mean(coverage_i))
            length_dict[r].append(np.mean(lengths_i))
            MSE_dict[r].append(MSE_test)

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