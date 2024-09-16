import numpy as np
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
from Utils.simulation_helpers import *
#from Utils.plotting import  *
import joblib
#sys.path.append('/home/yilingh/SI-CART')
def root_inference_sim(n=50, p=5, a=1, b=1,
                       sd_y=1, noise_sd=1, start=0, end=100):
    pivots = []
    naive_pivots = []
    for i in range(start, end):
        print(i, "th simulation")
        X = np.random.normal(size=(n, p))

        mu = b * ((X[:, 0] <= 0) * (1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
        y = mu + np.random.normal(size=(n,), scale=sd_y)
        # Create and train the regression tree
        reg_tree = RegressionTree(min_samples_split=10, max_depth=5, min_proportion=0.2)
        reg_tree.fit(X, y, sd=noise_sd)

        pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs = (
            reg_tree.condl_split_inference(node=reg_tree.root,
                                           ngrid=10000,
                                           ncoarse=200,
                                           grid_w_const=1.5,
                                           reduced_dim=1,
                                           sd=sd_y,
                                           use_cvxpy=True))

        target = norm_contrast.dot(mu)
        pivot_i = dist.ccdf(theta=target, x=obs_tar)
        pivots.append(pivot_i)

        naive_pivot = Z_test(y=y, norm_contrast=norm_contrast,
                             null=target)
        naive_pivots.append(naive_pivot)

    return pivots, naive_pivots

if __name__ == '__main__':
    # Get the script's directory
    #script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script's directory
    #os.chdir(script_directory)
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    argv = sys.argv
    ## sys.argv: [something, start, end, n, p, y_sd, omega_sd]
    start, end, n, p, sd_y, sd_noise = (int(argv[1]), int(argv[2]), int(argv[3]),
                                        int(argv[4]), int(argv[5]), float(argv[6]))

    pivots,naive_pivots = root_inference_sim(start=start, end=end, n=n, p=p,
                                sd_y=sd_y, noise_sd=sd_noise)

    #start, end, randomizer_scale, ncores = 0, 40, 1.5, 4
    dir = ('root_inference'
           + 'n' + str(n) + 'p' + str(p)
           + 'y' + str(sd_y) + 'o' + str(sd_noise) + '_'
           + str(start) + '_' + str(end) + '.pkl')
    joblib.dump([pivots,naive_pivots], dir, compress=1)