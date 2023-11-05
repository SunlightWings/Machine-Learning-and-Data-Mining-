# Decision Tree Classifier
This Python script implements a supervised machine learning algorithm known as the Decision Tree Classifier. The core idea behind this algorithm is to iteratively partition the dataset based on specific criteria (e.g., entropy, 
GINI index) to create a tree-like structure where the final nodes represent the target classes.

## Splitting Criteria
The splitting criteria intelligently selects attributes to form nodes, ensuring that the resulting child nodes are as pure as possible.

## Preventing Overfitting
To avoid overfitting, this implementation offers two pruning strategies:
    1. Pre-Pruning: This involves specifying a maximum depth for the tree, limiting its growth.
    2. Post-Pruning: Utilizes techniques like GridsearchCV and Cost-Complexity Pruning to trim the tree after it's been built.

## Note":
Prone to Overfitting: Decision trees can easily overfit noisy data. Pruning techniques and setting constraints on tree depth are essential to mitigate this.
Sensitive to Small Changes: A small change in the training data can result in a completely different tree structure.
