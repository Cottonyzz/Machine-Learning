import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=0.0001
max_iters=1000

def main():

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()

  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0

  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))

  logging.info("\n---------------------------------------------------------------------------\n")

  X_train_bias = dummyAugment(X_train)

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0

  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))


  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  logging.info("\n---------------------------------------------------------------------------\n")

  logging.info("Running cross-fold validation for bias case:")

  # Perform k-fold cross
  for k in [2,3,4, 5, 10, 20, 50]:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  ####################################################
  # Q8: Generate test predictions and write submission
  ####################################################

  # Retrain on full training set with bias, more iterations for better convergence
  X_train_bias_full = dummyAugment(X_train)
  w_final, _ = trainLogistic(X_train_bias_full, y_train, max_iters=5000, step_size=0.0001)

  # X_test has no label column (loadData returns it as-is)
  X_test_bias = dummyAugment(X_test)
  test_preds = (X_test_bias @ w_final >= 0).astype(int)

  with open('submission.csv', 'w') as f:
    f.write('id,type\n')
    for idx, pred in enumerate(test_preds.flatten()):
      f.write(f'{idx},{pred}\n')

  logging.info("Kaggle submission written to submission.csv")



######################################################################
# Q3.1 logistic
######################################################################
def logistic(z):
  logit_z = 1.0 / (1.0 + np.exp(-z))
  return logit_z


######################################################################
# Q3.2 calculateNegativeLogLikelihood
######################################################################
def calculateNegativeLogLikelihood(X,y,w):
  eps = 1e-7  # prevent log(0)
  probs = logistic(X @ w)                                          # n x 1
  nll = -np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
  return nll


######################################################################
# Q4 trainLogistic
######################################################################
def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):

    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) )

    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]

    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):

        # Gradient of negative log-likelihood: X^T (sigma(Xw) - y)
        w_grad = X.T @ (logistic(X @ w) - y)                      # d x 1

        # This is here to make sure your gradient is the right shape
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad

        losses.append(calculateNegativeLogLikelihood(X,y,w))

    return w, losses


######################################################################
# Q5 dummyAugment
######################################################################
def dummyAugment(X):
  ones = np.ones((X.shape[0], 1))
  aug_X = np.hstack([ones, X])
  return aug_X


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))

  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):

    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]

    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")

  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test

  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
