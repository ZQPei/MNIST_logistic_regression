import numpy as np

def sigmoid(matrix):
    return 1/(1+np.exp(-matrix))

def softmax(y_pred):
    sum = np.sum(y_pred,axis=1)
    return y_pred/sum[:,np.newaxis]

def predict(X_train, Theta):
    X = np.hstack((np.ones((X_train.shape[0],1)), X_train))
    y_pred =  sigmoid(np.dot(X, Theta)) # m X k
    return softmax(y_pred)

if __name__ == '__main__':
    print(sigmoid(np.array([-1,0,1,2])))