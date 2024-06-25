import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split on
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value for leaf nodes (mean of target values)

class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=float('inf')):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._get_best_split(X, y, num_features)
            if best_split["gain"] > 0:
                left_subtree = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right_subtree = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return TreeNode(feature_index=best_split["feature_index"], threshold=best_split["threshold"],
                                left=left_subtree, right=right_subtree)
        leaf_value = self._calculate_leaf_value(y)
        return TreeNode(value=leaf_value)

    def _get_best_split(self, X, y, num_features):
        best_split = {}
        max_gain = -float('inf')
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                if len(X_left) > 0 and len(X_right) > 0:
                    curr_gain = self._calculate_information_gain(y, y_left, y_right)
                    if curr_gain > max_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["X_left"] = X_left
                        best_split["y_left"] = y_left
                        best_split["X_right"] = X_right
                        best_split["y_right"] = y_right
                        best_split["gain"] = curr_gain
                        max_gain = curr_gain
        return best_split

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _calculate_information_gain(self, y, y_left, y_right):
        var_total = np.var(y) * len(y)
        var_left = np.var(y_left) * len(y_left)
        var_right = np.var(y_right) * len(y_right)
        return var_total - (var_left + var_right)

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._predict(sample, self.root) for sample in X])

    def _predict(self, sample, tree):
        if tree.value is not None:
            return tree.value
        feature_value = sample[tree.feature_index]
        if feature_value <= tree.threshold:
            return self._predict(sample, tree.left)
        else:
            return self._predict(sample, tree.right)

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9]])
    y = np.array([1, 2, 3, 4, 5, 6])

    # Create and train the regression tree
    reg_tree = RegressionTree(min_samples_split=2, max_depth=3)
    reg_tree.fit(X, y)

    # Make predictions
    predictions = reg_tree.predict(X)
    print(predictions)
