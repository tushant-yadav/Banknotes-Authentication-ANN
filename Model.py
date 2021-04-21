import numpy as np
from pandas import read_csv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from helper import *

def model(X, Y,TX,TY, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=True, AF= 'relu'):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    test_costs=[]
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters,AF)
        TL,_ = L_model_forward(TX, parameters, AF)
        # Compute cost.
        train_cost = compute_cost(AL, Y)
        test_cost = compute_cost(TL,TY)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches,AF)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("After iteration %i Training Cost :  %f  Testing Cost: %f " %(i, train_cost,test_cost))

        if print_cost and i % 100 == 0:
            costs.append(train_cost)
            test_costs.append(test_cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs),'-b',label='Training loss')
    plt.plot(np.squeeze(test_costs),'-r',label='Testing loss')  
    plt.legend(loc='upper right', frameon=False)
  
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
