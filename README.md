# CS 434 — Machine Learning Coursework

A collection of machine learning implementations from CS 434 (Machine Learning) at Oregon State University, built from scratch in NumPy without relying on high-level ML frameworks (except where explicitly permitted).

## hw1-knn/ — k-Nearest Neighbors

A vectorized kNN classifier used for both a Kaggle-style prediction task and k-fold cross validation to select the best `k`.

- `knn.py` — `get_nearest_neighbors`, `knn_classify_point`, `cross_validation`
- `train.csv`, `test_pub.csv` — training/test data
- `test_predicted.csv` — final Kaggle submission output

## hw2-logistic-regression/ — Logistic Regression for Tumor Diagnosis

Logistic regression trained with batch gradient descent (from scratch), including a bias-augmentation comparison and k-fold cross validation, applied to a breast tumor diagnosis dataset.

- `logistic.py` — `logistic`, `calculateNegativeLogLikelihood`, `trainLogistic`, `dummyAugment`, `kFoldCrossVal`
- `train_cancer.csv`, `test_cancer_pub.csv` — training/test data
- `submission.csv` — final Kaggle submission output

## hw4-neural-network/ — Feedforward Neural Network from Scratch (MNIST)

A fully from-scratch feedforward neural network (forward pass, backprop, SGD) trained on MNIST digit classification — no PyTorch/TensorFlow. Configurable depth, width, and activation function (Sigmoid/ReLU), trained with mini-batch SGD and cross-entropy loss.

- `nn.py` — full network implementation and training loop
- `writeup.pdf` — assignment writeup with results and analysis (93.8% validation accuracy)

## hw5-decision-trees-kmeans/ — Ensemble Methods and K-Means Clustering

Two parts: (1) measuring prediction correlation across decision tree ensembles under bagging and random feature subsampling (`max_features`), and (2) a from-scratch k-means clustering implementation applied to a toy 2D dataset and real image data (HOG features).

- `decision.py` — baseline vs. bagging vs. `max_features` ensemble correlation study, using `sklearn.tree.DecisionTreeClassifier` as the base learner (per assignment instructions)
- `kmeans.py` — `initalizeCentroids`, `computeAssignments`, `updateCentroids`, `calculateSSE` implemented in pure NumPy (no sklearn)
- Note: `img.npy` and `hog.npy` (the real image dataset used in `imageProblem()`) are course-provided binary data files and are not included here — `kmeans.py`'s `toyProblem()` function runs standalone without them.

## Running any of these

Each assignment is self-contained within its folder. Install dependencies and run the script directly:

```bash
pip install numpy matplotlib scikit-learn
cd hw1-knn && python knn.py
```
