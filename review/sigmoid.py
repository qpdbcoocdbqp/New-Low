import math

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1 + ez)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

print(sigmoid(0))             # 0.5
print(sigmoid_derivative(0))  # 0.25
print(sigmoid(-1000))         # 0.0,不會報錯
