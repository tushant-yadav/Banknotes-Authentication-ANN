# Banknotes-Authentication-ANN
Technical Report
Tushant Yadav
 

Code can be run without any installation using Google Colab. Link at last.
Github repository for code. Link at last.
The flow of code:
Random Initialization of the parameters for a 3 layer neural network. The dimension of neural network layers is [4,10,10,1].
forward propagation part
LINEAR part of a layer's forward propagation step.
Then we give the ACTIVATION function (relu/Leaky relu/sigmoid/tanh).
Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer L). 
This gives us a new L_model_forward function.
Compute the loss.
Implement the backward propagation module.
LINEAR part of a layer's backward propagation step.
We get the gradient of the ACTIVATE function (relu_backward, etc.)
Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
Finally, update the parameters. The weight matrix and biases are updated at every epoch.
The model's structure is [LINEAR -> RELU] x (L-1) -> LINEAR -> SIGMOID.


The repository contains 3 files:
Helper.py
This file consists of all the helper functions required for running of the model. For example activation functions, forward propagation function, etc.  
Model.py
This file contains the model and its structure once the model is called it uses helper functions to learn and predict.
main.py
This file contains all the data preprocessing and runs the model for various activation functions.


Results:
Sigmoid

After iteration 0 
Training Cost :  0.750625  
Testing Cost: 0.746729

After iteration 2900 
Training Cost :  0.349225  
Testing Cost: 0.368745 

training accuracy
Accuracy: 0.9260065288356907
test accuracy
Accuracy: 0.9183222958057398

F1 score: 0.895184
Leaky ReLu

After iteration 0 
Training Cost :  0.774012  
Testing Cost: 0.792092

After iteration 2900 
Training Cost :  0.012812  
Testing Cost: 0.024411 

training accuracy
Accuracy: 0.976060935799782
test accuracy
Accuracy: 0.9735099337748347

F1 score: 0.969697

ReLu

After iteration 0 
Training Cost :  0.780015  
Testing Cost: 0.797278

After iteration 2900 
Training Cost :  0.007311  
Testing Cost: 0.011961 

training accuracy
Accuracy: 0.9999999999999998
test accuracy
Accuracy: 0.9999999999999998

F1 score: 1.000000

Tanh

After iteration 0 
Training Cost :  0.581206  
Testing Cost: 0.601567

After iteration 2900 
Training Cost :  0.589392  
Testing Cost: 0.609148 

training accuracy
Accuracy: 0.44069640914036995
test accuracy
Accuracy: 0.43046357615894043

F1 score: 0.601852

Colab notebook link:- 
https://colab.research.google.com/drive/1XGXsLRaEvqBUVuKHb4a3cTd-DTXLkl04#scrollTo=sgDHQvrYV4V2
Github repository link:-
https://github.com/tushant-yadav/Banknotes-Authentication-ANN



