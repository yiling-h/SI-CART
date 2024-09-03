import numpy as np
from CART import RegressionTree
from Utils.plotting import  *
import time, sys, joblib, os
sys.path.append('/home/yilingh/SI-CART')
def root_inference_sim(n=50, p=5, a=1, b=1,
                       sd_y=2, noise_sd=1, start=0, end=100):
    pivots = []
    for i in range(start, end):
        print(i, "th simulation")
        X = np.random.normal(size=(n, p))

        mu = b * ((X[:, 0] <= 0) * (1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
        y = mu + np.random.normal(size=(n,), scale=sd_y)
        # Create and train the regression tree
        reg_tree = RegressionTree(min_samples_split=10, max_depth=5)
        reg_tree.fit(X, y, sd=noise_sd)

        pval, dist, contrast, norm_contrast, obs_tar, logW, suff, sel_probs = (
            reg_tree.condl_split_inference(node=reg_tree.root,
                                           ngrid=10000,
                                           ncoarse=100,
                                           grid_width=15,
                                           reduced_dim=3,
                                           sd=sd))

        target = norm_contrast.dot(mu)
        pivot_i = dist.ccdf(theta=target, x=obs_tar)
        pivots.append(pivot_i)

    return pivots

if __name__ == '__main__':
    # Get the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script's directory
    os.chdir(script_directory)
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    argv = sys.argv
    ## sys.argv: [something, start, end, randomizer_scale, ncores]
    start, end = int(argv[1]), int(argv[2])

    pivots = root_inference_sim(start=start, end=end)

    #start, end, randomizer_scale, ncores = 0, 40, 1.5, 4
    dir = ('root_inference' + str(start) + '_' + str(end) + '.pkl')
    joblib.dump(pivots, dir)