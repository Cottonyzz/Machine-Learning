import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
np.random.seed(102)
step_size = 0.01
batch_size = 200
max_epochs = 200

# GLOBAL PARAMETERS FOR NETWORK ARCHITECTURE
number_of_layers = 2
width_of_layers = 16  # only matters if number of layers > 1
activation = "ReLU" if False else "Sigmoid"

def main():

    # Load data and display an example
    X_train, Y_train, X_val, Y_val, X_test = loadData()
    displayExample(X_train[np.random.randint(0, len(X_train))])

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below
    net = FeedForwardNeuralNetwork(X_train.shape[1], 10, width_of_layers, number_of_layers, activation=activation)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # Loss function
    lossFunc = CrossEntropySoftmax()

    # Indices we will use to shuffle data randomly
    inds = np.arange(len(X_train))
    for i in range(max_epochs):

        # Shuffled indices so we go through data in new random batches
        np.random.shuffle(inds)

        # Go through all datapoints once (aka an epoch)
        j = 0
        acc_running = loss_running = 0
        while j < len(X_train):

            # Select the members of this random batch
            b = min(batch_size, len(X_train) - j)
            X_batch = X_train[inds[j:j+b]]
            Y_batch = Y_train[inds[j:j+b]].astype(int)

            # Compute the scores for our 10 classes using our model
            logits = net.forward(X_batch)
            loss = lossFunc.forward(logits, Y_batch)
            acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == Y_batch)

            # Compute gradient of Cross-Entropy Loss with respect to logits
            loss_grad = lossFunc.backward()

            # Pass gradient back through networks
            net.backward(loss_grad)

            # Take a step of gradient descent
            net.step(step_size)

            # Record losses and accuracy then move to next batch
            losses.append(loss)
            accs.append(acc)
            loss_running += loss * b
            acc_running += acc * b

            j += batch_size

        # Evaluate performance on validation
        vloss, vacc = evaluateValidation(net, X_val, Y_val, batch_size)
        val_losses.append(vloss)
        val_accs.append(vacc)

        logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(
            i, loss_running / len(X_train), acc_running / len(X_train) * 100, vacc * 100))

    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses, c="red", label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs, c="blue", label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.savefig("plot_default.png", dpi=100, bbox_inches='tight')
    plt.close()
    logging.info("Saved plot_default.png")

    ################################
    # Q5 Evaluate on Test
    ################################
    logging.info("Generating test set predictions...")
    test_logits = net.forward(X_test)
    test_preds = np.argmax(test_logits, axis=1)

    with open("kaggle_submission.csv", "w") as f:
        f.write("id,digit\n")
        for idx, pred in enumerate(test_preds):
            f.write(f"{idx},{pred}\n")
    logging.info("Saved kaggle_submission.csv")


class LinearLayer:

    # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.ones((1, output_dim)) * 0.5

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):
        self.input = input  # Storing X
        return self.input @ self.weights + self.bias

    #################################################
    # Q1 Implementing Backward Pass for Linear
    #################################################
    def backward(self, grad):  # grad is dL/dZ, shape (n, output_dim)
        self.grad_weights = self.input.T @ grad          # dL/dW  (n x d).T @ (n x c) = (d x c)  Eq.26
        self.grad_bias    = np.sum(grad, axis=0, keepdims=True)  # dL/db  sum rows -> (1 x c)   Eq.21
        return grad @ self.weights.T                     # dL/dX  (n x c) @ (c x d) = (n x d)  Eq.29

    def step(self, step_size):
        self.weights -= step_size * self.grad_weights
        self.bias    -= step_size * self.grad_bias


###############################################
# Instructor code below. Worth understanding
# but don't need to modify for base assignment
###############################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation="ReLU"):
        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            self.layers = [LinearLayer(input_dim, hidden_dim)]
            self.layers.append(Sigmoid() if activation == "Sigmoid" else ReLU())
            for i in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(Sigmoid() if activation == "Sigmoid" else ReLU())
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size=0.001):
        for layer in self.layers:
            layer.step(step_size)


# Sigmoid or Logistic Activation Function
class Sigmoid:
    def forward(self, input):
        self.act = 1 / (1 + np.exp(-input))
        return self.act

    def backward(self, grad):
        return grad * self.act * (1 - self.act)

    def step(self, step_size):
        return


# Rectified Linear Unit Activation Function
class ReLU:
    def forward(self, input):
        self.mask = (input > 0)
        return input * self.mask

    def backward(self, grad):
        return grad * self.mask

    def step(self, step_size):
        return


#####################################################
# Utility Functions for Computing Loss / Val Metrics
#####################################################
def softmax(x):
    x -= np.max(x, axis=1)[:, np.newaxis]  # Numerical stability trick
    return np.exp(x) / (np.sum(np.exp(x), axis=1)[:, np.newaxis])


class CrossEntropySoftmax:
    def forward(self, logits, labels):
        self.probs = softmax(logits)
        self.labels = labels
        return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:, np.newaxis], labels] + 0.00001))

    def backward(self):
        grad = self.probs
        grad[np.arange(len(self.probs))[:, np.newaxis], self.labels] -= 1
        return grad.astype(np.float64) / len(self.probs)


def evaluateValidation(model, X_val, Y_val, batch_size):
    val_loss_running = 0
    val_acc_running = 0
    j = 0
    lossFunc = CrossEntropySoftmax()
    while j < len(X_val):
        b = min(batch_size, len(X_val) - j)
        X_batch = X_val[j:j+b]
        Y_batch = Y_val[j:j+b].astype(int)
        logits = model.forward(X_batch)
        loss = lossFunc.forward(logits, Y_batch)
        acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == Y_batch)
        val_loss_running += loss * b
        val_acc_running += acc * b
        j += batch_size
    return val_loss_running / len(X_val), val_acc_running / len(X_val)


#####################################################
# Utility Functions for Loading and Displaying Data
#####################################################
def loadData(normalize=True):
    train = np.loadtxt("mnist_small_train.csv", delimiter=",", dtype=np.float64)
    val   = np.loadtxt("mnist_small_val.csv",   delimiter=",", dtype=np.float64)
    test  = np.loadtxt("mnist_small_test.csv",  delimiter=",", dtype=np.float64)

    if normalize:
        X_train = train[:, :-1] / 256 - 0.5
        X_val   = val[:,   :-1] / 256 - 0.5
        X_test  = test / 256 - 0.5
    else:
        X_train = train[:, :-1]
        X_val   = val[:,   :-1]
        X_test  = test

    Y_train = train[:, -1].astype(int)[:, np.newaxis]
    Y_val   = val[:,  -1].astype(int)[:, np.newaxis]

    logging.info("Loaded train: " + str(X_train.shape))
    logging.info("Loaded val: "   + str(X_val.shape))
    logging.info("Loaded test: "  + str(X_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test


def displayExample(x):
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.savefig("example_digit.png", dpi=80, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
