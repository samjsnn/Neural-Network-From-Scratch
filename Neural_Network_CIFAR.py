import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore")


# Testing with a small neural network

print("Testing with a small neural network")


class NeuralNetwork:
    def __init__(self):
        # Define the architecture of the neural network
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 2

        # Initialize the weights and biases
        self.weights1 = np.array([[0.1, 0.1], [0.2, 0.1]])
        self.bias1 = np.array([0.1, 0.1])
        self.weights2 = np.array([[0.1, 0.1], [0.1, 0.2]])
        self.bias2 = np.array([0.1, 0.1])

        # Initialize the learning rate
        self.learning_rate = 0.1

    def forward(self, X):
        # Perform the forward pass
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights2) + self.bias2

    def backward(self, X, y):
        # Compute the gradients
        output_error = self.output - y
        hidden_error = np.dot(output_error, self.weights2.T) * self.sigmoid_derivative(
            self.hidden_input
        )

        # Update the weights and biases
        self.weights2 -= self.learning_rate * \
            np.dot(self.hidden_output.T, output_error)
        self.bias2 -= self.learning_rate * np.sum(output_error, axis=0)
        self.weights1 -= self.learning_rate * np.dot(X.T, hidden_error)
        self.bias1 -= self.learning_rate * np.sum(hidden_error, axis=0)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        self.forward(X)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


# Create an instance of the neural network
model = NeuralNetwork()

# Define the training samples and their labels
X = np.array([[0.1, 0.1], [0.1, 0.2]])
y = np.array([[1, 0], [0, 1]])

# Train the neural network
model.train(X, y, epochs=1)

# Print the updated weights and biases
print("Updated Weights and Biases:")
print("Weights1:")
print(model.weights1)
print("Bias1:")
print(model.bias1)
print("Weights2:")
print(model.weights2)
print("Bias2:")
print(model.bias2)
print("\n")


# Loading Data

print("Loading Data")


def load_cifar_batch(file):
    with open(file, "rb") as fo:
        batch = pickle.load(fo, encoding="bytes")
    data = batch[b"data"]
    labels = batch[b"labels"]
    data = data.reshape(-1, 3, 32, 32).transpose(
        0, 2, 3, 1
    )  # Reshape and transpose dimensions
    return data, labels


# Load training batches

print("Loading training batches")
train_data = []
train_labels = []
for i in range(1, 6):
    file = f"data_cifar10/data_batch_{i}"
    data, labels = load_cifar_batch(file)
    train_data.append(data)
    train_labels += labels

X_train = np.concatenate(train_data, axis=0)
y_train = np.array(train_labels)

# Load test batch

print("Loading test batch")
X_test, test_labels = load_cifar_batch("data_cifar10/test_batch")


# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train / 255.0
X_test = X_test / 255.0


def to_categorical(labels, num_classes):
    categorical_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        categorical_labels[i, label] = 1
    return categorical_labels


# Convert labels to categorical format
num_classes = 10
y_train = np.array(to_categorical(train_labels, num_classes))
y_test = np.array(to_categorical(test_labels, num_classes))


# flatten the input data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# Display the shapes of the loaded data
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)

print("Test labels shape:", y_test.shape)


# Neural Network Class
class NeuralNetwork:
    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        return -(y_true * np.log(y_pred)).sum()

    @staticmethod
    def accuracy(y_pred, y_true):
        return np.sum(y_pred == y_true)

    @staticmethod
    def softmax(x):
        expx = np.exp(x - np.max(x, axis=1, keepdims=True))
        return expx / expx.sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, input_size, hidden_nodes, output_size, mode):
        self.num_layers = 3
        self.input_shape = input_size
        self.hidden_shape = hidden_nodes
        self.output_shape = output_size
        self.mode = mode

        self.weights_ = self.biases_ = []
        self.__init_weights()

    def __init_weights(self):
        W_h = np.random.normal(size=(self.input_shape, self.hidden_shape))
        b_h = np.zeros(shape=(self.hidden_shape,))

        W_o = np.random.normal(size=(self.hidden_shape, self.output_shape))
        b_o = np.zeros(shape=(self.output_shape,))

        self.weights_ = [W_h, W_o]
        self.biases_ = [b_h, b_o]

    def fit(self, Xs, Ys, epochs, lr=1e-3, batch_size=32):
        history = []
        num_samples = Xs.shape[0]
        num_batches = num_samples // batch_size

        for epoch in (range(epochs)):
            indices = np.random.permutation(num_samples)
            Xs = Xs[indices]
            Ys = Ys[indices]

            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                batch_X = Xs[start:end]
                batch_Y = Ys[start:end]

                activations = self.forward_pass(batch_X)
                deltas = self.backward_pass(batch_Y, activations)

                layer_inputs = [batch_X] + activations[:-1]
                self.weight_update(deltas, layer_inputs, lr)

            preds = self.predict(Xs)
            current_loss = self.cross_entropy_loss(preds, Ys)
            history.append(current_loss)

        return np.array(history) / len(Xs)

    def forward_pass(self, input_data):
        activations = []

        z_1 = input_data.dot(self.weights_[0]) + self.biases_[0]
        a_1 = self.sigmoid(z_1)
        z_2 = a_1.dot(self.weights_[1]) + self.biases_[1]
        a_2 = []

        a_2 = self.softmax(z_2)

        activations.append(a_1)
        activations.append(a_2)

        return activations

    def backward_pass(self, targets, layer_activations):
        deltas = []
        a_1 = layer_activations[0]
        a_2 = layer_activations[1]

        phi_2 = a_2 - targets
        deriv_2 = a_2 * (1 - a_2)
        del_2 = np.multiply(phi_2, deriv_2)

        phi_1 = np.transpose(self.weights_[1].dot(np.transpose(del_2)))
        deriv_1 = a_1 * (1 - a_1)
        del_1 = np.multiply(phi_1, deriv_1)

        deltas.append(del_1)
        deltas.append(del_2)

        return deltas

    def weight_update(self, deltas, layer_inputs, lr):
        a_1 = layer_inputs[1]
        x = layer_inputs[0]
        del_1 = deltas[0]
        del_2 = deltas[1]

        self.weights_[1] -= lr * np.transpose(np.transpose(del_2).dot(a_1))
        self.biases_[1] -= lr * np.sum(del_2, axis=0)

        self.weights_[0] -= lr * np.transpose(np.transpose(del_1).dot(x))
        self.biases_[0] -= lr * np.sum(del_1, axis=0)

    def predict(self, Xs):
        predictions = []
        num_samples = Xs.shape[0]
        for i in range(num_samples):
            sample = Xs[i, :].reshape((1, self.input_shape))
            sample_prediction = self.forward_pass(sample)[-1]
            predictions.append(sample_prediction.reshape((self.output_shape,)))
        return np.array(predictions)

    def evaluate(self, Xs, Ys):
        pred = self.predict(Xs)
        return self.cross_entropy_loss(pred, Ys), self.accuracy(
            pred.argmax(axis=1), Ys.argmax(axis=1)
        )


# Create and train the model
print("Training the model...")
nn = NeuralNetwork(
    input_size=3072, hidden_nodes=30, output_size=num_classes, mode="classification"
)
loss = nn.fit(Xs=X_train, Ys=y_train, epochs=20, lr=0.1, batch_size=100)


# Calculating loss and accuracy
print("Calculating loss and accuracy")

cross_entropy, accuracy = nn.evaluate(X_test, y_test)
print("Loss", cross_entropy)


print("Accuracy : {}".format((accuracy / X_test.shape[0]) * 100))


# Learning rate experiments
print("Learning rate experiments...")

learning_rates = [0.001, 0.01, 1.0, 10.0, 100.0]
accuracies = []
for learning_rate in learning_rates:
    nn1 = NeuralNetwork(
        input_size=3072, hidden_nodes=30, output_size=num_classes, mode="classification"
    )
    nn1.fit(Xs=X_train, Ys=y_train, epochs=20,
            lr=learning_rate, batch_size=100)
    _, acc = nn1.evaluate(X_test, y_test)
    accuracies.append(acc)


accuracies = [(i / X_test.shape[0]) * 100 for i in accuracies]


print(
    "maximum accuracy achieved using different learning rates: ",
    np.round(np.max(accuracies), 2),
    " using learning rate :",
    learning_rates[np.argmax(accuracies)],
)
print(
    "minimum accuracy achieved using different learning rates: ",
    np.round(np.min(accuracies), 2),
    " using learning rate :",
    learning_rates[np.argmin(accuracies)],
)


plt.plot(learning_rates, accuracies)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracies")
plt.show()


# Mini-batch size experiments

print("Mini-batch size experiments")
batches = [1, 5, 20, 100, 300]
accuracies = []
for batch in batches:
    nn1 = NeuralNetwork(
        input_size=3072, hidden_nodes=30, output_size=num_classes, mode="classification"
    )
    nn1.fit(Xs=X_train, Ys=y_train, epochs=20, lr=0.1, batch_size=batch)
    _, acc = nn1.evaluate(X_test, y_test)
    accuracies.append(acc)


accuracies = [(i / X_test.shape[0]) * 100 for i in accuracies]


print(
    "maximum accuracy achieved using different learning rates: ",
    np.round(np.max(accuracies), 2),
    " using  mini-batch size:",
    learning_rates[np.argmax(accuracies)],
)
print(
    "minimum accuracy achieved using different learning rates: ",
    np.round(np.min(accuracies), 2),
    " using  mini-batch size :",
    learning_rates[np.argmin(accuracies)],
)


plt.plot(learning_rates, accuracies)
plt.xlabel("Batch Size")
plt.ylabel("Accuracies")
plt.show()
