import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

path = "img"
if not os.path.exists(os.path.join(os.getcwd(),path)):
    os.mkdir(path)

with open("theta","r") as theta:
    Theta = np.fromfile(theta).reshape(785,10)

Weight = Theta[1:,:].transpose() # 10 X 784



for i in range(10):
    w = Weight[i].reshape(28,28)
    #w = w/np.max(w)
    #cv2.imshow("%d"%i, w)

    #plt.figure(i)
    plt.imshow(w)
    plt.savefig("img/%d"%i)

#plt.show()