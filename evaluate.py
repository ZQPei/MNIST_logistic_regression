import numpy as np
import matplotlib.pyplot as plt
import cv2

import os

from load_mnist import load_mnist
from predict import predict

with open("theta","r") as theta:
    Theta = np.fromfile(theta).reshape(785,10)

path = '/opt/data/pzq/MNIST'
X_test, y_test = load_mnist(path,mode = 'test')
y_pred_ori = predict(X_test, Theta) # m X k
y_pred = np.argmax(y_pred_ori, axis=1) # m X 1

# visualize
idx = 2345
for i in range(10):
    

    y_pred_idx = y_pred_ori[idx]
    y = y_test[idx]

    print(idx)
    print(y_pred_idx)
    print(y)


    img = X_test[idx].reshape(28,-1)
    cv2.imshow("digit",img)
    idx += 100
    cv2.waitKey()
    #cv2.destroyAllWindows()

acc = np.sum(np.abs(y_pred-y_test) == 0)/y_pred.size
print("acc = %f"%acc)