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

def pred_sim(start, end, n_test=50, n=50, p=5,
             sd_y=1.5, noise_sd=0.5, a=1, b=1, prune_level=0.1):
    unpruned_RMSE_test = []
    pruned_RMSE_test = []
    mean_RMSE_test = []
    unpruned_RMSE_train = []
    pruned_RMSE_train = []
    mean_RMSE_train = []
    for i in range(start, end):
        print(i, "th simulation")
        # Generate training set
        X = np.random.normal(size=(n, p))
        mu = b * ((X[:, 0] <= 0) * (1 + a * (X[:, 1] > 0) + (X[:, 2] * X[:, 1] <= 0)))
        sd = sd_y
        y = mu + np.random.normal(size=(n,), scale=sd)
        # Create and train the regression tree
        reg_tree = RegressionTree(min_samples_split=10, max_depth=3, min_proportion=0.2)
        reg_tree.fit(X, y, sd=noise_sd)

        # Generate testing set
        X_test = np.random.normal(size=(n_test, p))
        mu_test = b * ((X_test[:, 0] <= 0) * (1 + a * (X_test[:, 1] > 0)
                                              + (X_test[:, 2] * X_test[:, 1] <= 0)))
        y_test = mu_test + np.random.normal(size=(n_test,), scale=sd)
        pred_test = reg_tree.predict(X_test)
        pred_train = reg_tree.predict(X)
        # Evaluate unpruned test error
        RMSE_unpruned_train = np.sqrt(np.mean((y - pred_train) ** 2))
        RMSE_unpruned_test = np.sqrt(np.mean((y_test - pred_test) ** 2))
        reg_tree.bottom_up_pruning(level=prune_level)

        # Evaluate pruned test error
        pruned_pred_train = reg_tree.predict(X)
        pruned_pred_test = reg_tree.predict(X_test)
        RMSE_pruned_train = np.sqrt(np.mean((y - pruned_pred_train) ** 2))
        RMSE_pruned_test = np.sqrt(np.mean((y_test - pruned_pred_test) ** 2))
        naive_RMSE_train = np.std(y)
        naive_RMSE_test = np.sqrt(np.mean((np.ones((n_test,)) * np.mean(y) - y_test) ** 2))

        unpruned_RMSE_train.append(RMSE_unpruned_train)
        pruned_RMSE_train.append(RMSE_pruned_train)
        mean_RMSE_train.append(naive_RMSE_train)
        unpruned_RMSE_test.append(RMSE_unpruned_test)
        pruned_RMSE_test.append(RMSE_pruned_test)
        mean_RMSE_test.append(naive_RMSE_test)

    return (unpruned_RMSE_train, pruned_RMSE_train, mean_RMSE_train,
            unpruned_RMSE_test, pruned_RMSE_test, mean_RMSE_test)

if __name__ == '__main__':

    argv = sys.argv
    ## sys.argv: [something, start, end, n, p, y_sd, omega_sd]
    start, end, n, p, sd_y, sd_noise = (int(argv[1]), int(argv[2]), int(argv[3]),
                                        int(argv[4]), float(argv[5]), float(argv[6]))

    (unpruned_RMSE_train, pruned_RMSE_train, mean_RMSE_train,
     unpruned_RMSE_test, pruned_RMSE_test, mean_RMSE_test)\
        = pred_sim(start=start, end=end,
                   n_test=50, n=n, p=p, sd_y=sd_y, noise_sd=sd_noise,
                   a=1, b=1, prune_level=0.1)

    #start, end, randomizer_scale, ncores = 0, 40, 1.5, 4
    dir = ('pruning_'
           + 'n' + str(n) + 'p' + str(p)
           + 'y' + str(sd_y) + 'o' + str(sd_noise) + '_'
           + str(start) + '_' + str(end) + '.pkl')
    joblib.dump([unpruned_RMSE_train, pruned_RMSE_train, mean_RMSE_train,
                 unpruned_RMSE_test, pruned_RMSE_test, mean_RMSE_test], dir, compress=1)