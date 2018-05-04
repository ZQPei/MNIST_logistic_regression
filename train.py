import numpy as np

def sigmoid(matrix):
    return 1./(1.+np.exp(-matrix))

def cost_function(y_train, X_train, Theta, reg_mode = True, l = 0.1):
    m,n = X_train.shape
    k = k = np.unique(y_train).size

    X = np.hstack((np.ones((m,1)), X_train))  # m X n+1

    h = sigmoid(np.dot(X, Theta))  #m X k
    y = np.zeros((m,k))            #m X k
    for i in range(m):
        y[i,y_train[i]] = 1
    
    cost = -1*np.sum(np.multiply(y,np.log(h))+np.multiply((1-y),np.log(1-h)), axis=0)/m  # k
    grad = np.dot(X.transpose(), (h-y))/m # n+1 X k

    if reg_mode:
        cost += l*np.sum(np.multiply(Theta[1:,:], Theta[1:,:]), axis=0)/(2*m)
        grad[1:,:] += l*Theta[1:,:]/m  # n X k
    
    #print(cost)
    #print(grad[0])
    return cost, grad

def gradient_descent(y_train, X_train, Theta, alpha, rounds = 400):
    m,n = X_train.shape
    k = np.unique(y_train).size
    cost_history = np.zeros((rounds,k)) # rounds * k

    for i in range(rounds):
        cost_history[i], grad = cost_function(y_train, X_train, Theta)
        Theta = Theta - alpha * grad
        print("Round %d , cost = "%(i), cost_history[i])
        
        
        if i>=1 :
            tmp = cost_history[i] >= cost_history[i-1]
            if tmp.any():
                loc = np.where(tmp)[0]
                alpha[loc] = alpha[loc]/3
            
    return cost_history, Theta
