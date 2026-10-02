import numpy as np
import time


def main():
    #############################################################
    # These first bits are just to help you develop your code
    # and have expected outputs given. All asserts should pass.
    #############################################################

    example_train_x = np.array([[ 1, 0, 2], [3, -2, 4], [5, -2, 4],
                                 [ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_y = np.array([[0], [1], [1], [1], [0], [1]])

    # Sanity Check 1: k=1 query with training points should return itself
    for i in range(len(example_train_x)):
        assert([i] == get_nearest_neighbors(example_train_x, example_train_x[i], 1))

    # Sanity Check 2: check neighbors for specific queries
    nn_idx = get_nearest_neighbors(example_train_x, np.array([1, 4, 2]), 2)
    assert(set(nn_idx).difference(set([4, 3])) == set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array([1, -4, 2]), 3)
    assert(set(nn_idx).difference(set([1, 0, 2])) == set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array([10, 40, 20]), 5)
    assert(set(nn_idx).difference(set([4, 3, 0, 2, 1])) == set())

    # Sanity Check 3: neighbors for increasing k should be subsets
    query = np.array([10, 40, 20])
    p_nn_idx = get_nearest_neighbors(example_train_x, query, 1)
    for k in range(2, 7):
        nn_idx = get_nearest_neighbors(example_train_x, query, k)
        assert(set(p_nn_idx).issubset(nn_idx))
        p_nn_idx = nn_idx

    # Test prediction
    queries = np.array([[10, 40, 20], [-2, 0, 5], [0, 0, 0]])
    pred = predict(example_train_x, example_train_y, queries, 3)
    assert(np.all(pred == np.array([[0], [1], [0]])))

    # Test accuracy
    true_y = np.array([[0], [1], [2], [1], [1], [0]])
    pred_y = np.array([[5], [1], [0], [0], [1], [0]])
    assert(compute_accuracy(true_y, pred_y) == 3/6)

    pred_y = np.array([[5], [1], [2], [0], [1], [0]])
    assert(compute_accuracy(true_y, pred_y) == 4/6)

    print("All sanity checks passed!")

    #######################################
    # Now on to the real data!
    #######################################
    train_X, train_y, test_X = load_data()

    #######################################
    # Q9 Hyperparameter Search
    #######################################
    print("Performing 4-fold cross validation")
    for k in [1, 3, 5, 7, 9, 99, 999, 8000]:
        t0 = time.time()

        # Compute train accuracy using the whole training set
        train_pred = predict(train_X, train_y, train_X, k)
        train_acc = compute_accuracy(train_y, train_pred)

        # Compute 4-fold cross validation accuracy
        val_acc, val_acc_var = cross_validation(train_X, train_y, num_folds=4, k=k)

        t1 = time.time()
        print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(
            k, train_acc * 100, val_acc * 100, val_acc_var * 100, t1 - t0))

    #######################################
    # Q10 Kaggle Submission
    #######################################
    # Set your best k value here after reviewing Q9 results
    best_k = 9

    pred_test_y = predict(train_X, train_y, test_X, best_k)

    test_out = np.concatenate((
        np.expand_dims(np.array(range(2000), dtype=int), axis=1),
        pred_test_y
    ), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')
    print("Test predictions saved to test_predicted.csv")


######################################################################
# Q7 get_nearest_neighbors
######################################################################
# Given an n-by-d matrix of training examples and a query point,
# returns the indices of the k nearest neighbors using Euclidean distance.
#
# The key trick: subtract the query from every row of example_set using
# numpy broadcasting, then take the L2 norm of each row. This gives all
# distances in one vectorized operation instead of looping.
######################################################################

def get_nearest_neighbors(example_set, query, k):
    # example_set - query broadcasts: subtracts query from every row
    # np.linalg.norm with axis=1 takes the L2 norm of each row (each difference vector)
    distances = np.linalg.norm(example_set - query, axis=1)

    # np.argsort returns the indices that would sort the array in ascending order
    # slicing [:k] gives us just the k smallest (i.e. k nearest)
    idx_of_nearest = np.argsort(distances)[:k]
    return idx_of_nearest


######################################################################
# Q7 knn_classify_point
######################################################################
# Runs a kNN classifier on a single query point.
# Gets the k nearest neighbors then returns whichever class label
# appears most among them (majority vote).
######################################################################

def knn_classify_point(examples_X, examples_y, query, k):
    # Get indices of the k nearest training points
    neighbor_indices = get_nearest_neighbors(examples_X, query, k)

    # Look up their labels
    neighbor_labels = examples_y[neighbor_indices]

    # Return the majority label using np.bincount
    # np.bincount counts occurrences of each non-negative integer
    # np.argmax returns the index (i.e. label) with the highest count
    predicted_label = np.argmax(np.bincount(neighbor_labels.flatten().astype(int)))
    return predicted_label


######################################################################
# Q8 cross_validation
######################################################################
# Performs K-fold cross validation for a kNN classifier.
# Splits the data into num_folds equal chunks, then for each fold
# trains on the other chunks and evaluates on the held-out fold.
# Returns the mean and variance of the per-fold accuracies.
######################################################################

def cross_validation(train_X, train_y, num_folds=4, k=1):
    # Split data into num_folds equal chunks along the row axis
    X_folds = np.split(train_X, num_folds)
    y_folds = np.split(train_y, num_folds)

    accuracies = []

    for i in range(num_folds):
        # Hold out fold i as the validation set
        val_X = X_folds[i]
        val_y = y_folds[i]

        # Stack all other folds together as the training set
        train_X_cv = np.vstack([X_folds[j] for j in range(num_folds) if j != i])
        train_y_cv = np.vstack([y_folds[j] for j in range(num_folds) if j != i])

        # Predict on the validation fold and compute accuracy
        pred_y = predict(train_X_cv, train_y_cv, val_X, k)
        acc = compute_accuracy(val_y, pred_y)
        accuracies.append(acc)

    avg_val_acc = np.mean(accuracies)
    varr_val_acc = np.var(accuracies)
    return avg_val_acc, varr_val_acc


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy


def predict(examples_X, examples_y, queries_X, k):
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]
    return np.array(predicted_y, dtype=int)[:, np.newaxis]


def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:, np.newaxis]

    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X


if __name__ == "__main__":
    main()
