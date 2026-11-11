import math, random

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1 + ez)

random.seed(0)
W1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]  # W1[j][i]
b1 = [0.0, 0.0]
W2 = [random.uniform(-1, 1) for _ in range(2)]
b2 = 0.0
lr = 0.5

data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]  # XOR

for epoch in range(10000):
    for x, y in data:
        # forward
        h = [sigmoid(sum(W1[j][i] * x[i] for i in range(2)) + b1[j]) for j in range(2)]
        o = sigmoid(sum(W2[j] * h[j] for j in range(2)) + b2)

        # backward
        d_o = (o - y) * o * (1 - o)
        d_h = [d_o * W2[j] * h[j] * (1 - h[j]) for j in range(2)]

        # update
        for j in range(2):
            W2[j] -= lr * d_o * h[j]
            for i in range(2):
                W1[j][i] -= lr * d_h[j] * x[i]
            b1[j] -= lr * d_h[j]
        b2 -= lr * d_o

for x, y in data:
    h = [sigmoid(sum(W1[j][i] * x[i] for i in range(2)) + b1[j]) for j in range(2)]
    o = sigmoid(sum(W2[j] * h[j] for j in range(2)) + b2)
    print(f"Input: {x}, Predicted: {o:.4f}, Actual: {y}")