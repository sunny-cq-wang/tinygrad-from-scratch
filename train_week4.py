# MNIST
import numpy as np

from engine.tensor import Tensor
from nn.Module import Module, Linear
from nn.losses import softmax_cross_entropy, to_one_hot


class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.fc3(x)


class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


# ----------------------------------------------------------------

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)

batch_size = 32
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(x_train), batch_size):
        x_batch = Tensor(x_train[i:i+batch_size])
        y_batch = to_one_hot(y_train[i:i+batch_size])

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = softmax_cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    print(f"Epoch {epoch+1}, Loss: {total_loss / (len(x_train) // batch_size):.4f}")


# --------------------------------------------------------------------

x_test_tensor = Tensor(x_test)
logits = model(x_test_tensor)
preds = np.argmax(logits.data, axis=1)
accuracy = np.mean(preds == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")