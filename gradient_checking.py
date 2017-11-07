'''
Created on Nov 7, 2017

@author: udit.gupta
'''
# Packages
from gradient_c_utils import *

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    print('Shape of parameter values is ' + str(parameters_values.shape))
    #print('Parameter values are :')
    #print(parameters_values)
    grad = gradients_to_vector(gradients)
    print('Shape of grads values is '+ str(grad.shape))
    #print('Gradients are :')
    #print(grad)
    num_parameters = parameters_values.shape[0]
    print('Number of parameters are : '+ str(num_parameters))
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    print('Shape of Jplus,Jminus and gradapprox is' + str(J_plus.shape))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        thetaplus = np.copy(parameters_values)                                      # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                # Step 2
        J_plus[i], _ = forward_propagation(X,Y,vector_to_dictionary(thetaplus))                                   # Step 3

        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)                                     # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                               # Step 2        
        J_minus[i], _ = forward_propagation(X,Y,vector_to_dictionary(thetaminus))                                  # Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                               # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                             # Step 2'
    difference = numerator / denominator                                          # Step 3'

    if difference > 2e-7:
        print ("There is a mistake in the backward propagation! difference = " + str(difference))
    else:
        print ("Your backward propagation works perfectly fine! difference = " + str(difference))
    
    return difference

train_X, train_Y, test_X, test_Y = load_2D_dataset()

layers_dims = [train_X.shape[0], 20, 3, 1]
parameters = initialize_parameters(layers_dims)

cost, cache = forward_propagation(train_X, train_Y, parameters)
gradients = backward_propagation(train_X, train_Y, cache)
difference = gradient_check_n(parameters, gradients, train_X, train_Y)
