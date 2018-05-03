import numpy as np

from load_mnist import load_mnist
from train import gradient_descent

path = '/opt/data/pzq/MNIST'
X_train, y_train = load_mnist(path)
m,n = X_train.shape
k = np.unique(y_train).size

# Initialize parameters
Theta = np.random.randn(n+1,k)  # n+1 X k

# Normalization
X_train = X_train/255

# train
alpha = np.ones(k).astype(np.float) * 3 # k
cost, Theta = gradient_descent(y_train,X_train,Theta, alpha)

with open("theta", "w") as theta:
    Theta.tofile(theta)

