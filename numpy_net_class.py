import numpy as np
import random as random
import math as math
import copy

class numpy_net(object):
    
    # initializer: get size of layers (including input and output), and activations
    def __init__(self, layers, layer_activation, output_activation, loss_function, cost_function):
        
        '''
        ATTRIBUTES
            # inputs and outputs
            a0 = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m training examples
            af = numpy array of predictions, must be (m,d,1) numpy array for output layer of d neurons with m training examples
            y = expected outputs (for cost/loss function), must be (m,d,1) numpy array for d outputs with m examples

            # network properties (weights, biases)
            weights = list of numpy arrays with weights for each layer
            biases = list of numpy arrays with biases for each layer
            nlayers = number of layers
            layers = number of neurons in each layer

            # activation and loss functions
            g_func = activation function tag for hidden layers
            h_func = activation function tag for output layer
            L_func = loss function tag
            J_func = cost function tag
            g = master function with all activation functions
            dg = master function with all activation function derivatives
            L = master function with all loss functions
            dL = master function with derivatives of loss functions
            J = master function with all cost functions

            # intermediates: activations of hidden layers, arrays used in gradient calculations
            A = list of activations of each layer
                0th element = (m,d,1) numpy array of activations of input layer, 
                1st element = (m,d,1) numpy array of activations of first hidden layer, etc
            Z = list of values Wa + B for each layer
                0th element = (m,d,1) numpy array of W_1a_0 + B_1 for a0 the (m,d,1) array of inputs, 
                1st element = (m,d1,1) numpy array of W_2a_1 + B_2 for a1 the (m,d1,1) array of first hidden layer activations, etc
            α = list of matrices (α_m)_ijk = (W_m)_jk * dg(q_ik) for q = z^(m-1)
                0th element = W_2 * dg(z_1), 1st element = W_3 * dg(z_2), etc
            Λ = list of numpy arrays Λ_(n-m) = α_n α_(n-1) ... α_(n-m+1)
                0th element = Λ_1, 1st element = Λ_2, etc
                i.e. 1st element = associated with gradients of 1st layer weights, 2nd element with 2nd layer weights, etc     
            η = (m,d,1) numpy array of dL(a_n) * dh(z_n) (element-wise)        

            # gradients
            dJdW = list of numpy arrays with gradients relative to weights of each layer
                dJdW[0] = numpy array w/ (ij)th element = gradient of cost J relative to (ij)th element of 1st layer weights
            dJdB = list of numpy arrays with gradients relative to biases of each layer
                dJdB[0] = numpy array w/ (i)th element = gradient of cost J relative to (i)th element of 1st layer biases    

            # learning parameters
            l_rate = learning rate for gradient descent (oops i already used α)
        '''            
        
        # initialize weights, biases with random values (-1,1), normalized by input of layer
        self.layers = layers
        self.nlayers = len(layers)
        self.weights = []
        self.biases = []
        for idx in range(self.nlayers - 1):
            
            # norm = layers[idx]
            # self.weights.append(np.random.rand(layers[idx+1], layers[idx]) / norm)
            # self.biases.append(np.random.rand(layers[idx+1], 1) / norm)
            
            (self.weights).append(np.random.normal(0.0, 1/np.sqrt(2 * layers[idx]), (layers[idx+1], layers[idx])))
            (self.biases).append(np.random.normal(0.0, 1/np.sqrt(2 * layers[idx]), (layers[idx+1], 1)))
            
        # input, output, and expected output (set later via methods)
        self.train_input = None
        self.test_input = None
        self.train_output = None
        
        # subsets of input, output for training
        self.a0 = None
        self.y = None
        
        # set tags for functions
        self.g_func = layer_activation
        self.h_func = output_activation
        self.L_func = loss_function
        self.J_func = cost_function
        
        # intermediate values (set later via methods)
        self.A = None
        self.Z = None
        self.α = None
        self.Λ = None        
        self.η = None
        
        # gradients
        self.dJdW = None
        self.dJdB = None
        
        # learning parameters (give a default rate)
        self.l_rate = 0.1

    '''
    PROPAGATION AND GRADIENT METHODS
    calculate_layer_activations = calculate activations and z = Wa + B for all layers, including hidden layers
    calculate_α = calculate intermediate matrices α for gradient calculation
    calculate_Λ = calculate intermediate matrices Λ for gradient calculation
    calculate_η = calculate intermediate matrices η for gradient calculation
    calculate_W_gradient = calculate dJdW only from Λ, η
    calculate_B_gradient = calculate dJdB only from Λ, η
    calculate_derivatives = calculate dJdW and dJdB from Λ, η
    calculate_gradJ = calculate dJdW and dJdB from start to finish
    '''    
        
    def calculate_layer_activations(self):

        '''
        INPUTS
        a0 = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m training examples
        weights = list of numpy arrays with weights for each layer
        biases = list of numpy arrays with biases for each layer
        g_func = activation function tag for hidden layers
        h_func = activation function tag for output layer
        g = master function with all activation functions

        SETS VALUES OF
        A = list of activations of each layer
            0th element = (m,d,1) numpy array of activations of input layer, 
            1st element = (m,d,1) numpy array of activations of first hidden layer, etc
        Z = list of values Wa + B for each layer
            0th element = (m,d,1) numpy array of W_1a_0 + B_1 for a0 the (m,d,1) array of inputs, 
            1st element = (m,d1,1) numpy array of W_2a_1 + B_2 for a1 the (m,d1,1) array of first hidden layer activations, etc

        NB: length of Z will be one less than length of A
        '''

        # initialize outputs
        self.A = [np.copy(self.a0)]
        self.Z = []

        # copy a0 to avoid overwriting
        t_a0 = np.copy(self.a0)
        
        for idx in range(len(self.weights)-1):

            # calculate Wa + B
            t_a0 = np.matmul(self.weights[idx], t_a0) + self.biases[idx]

            # save z
            self.Z.append(np.copy(t_a0))
            
            # apply activation
            t_a0 = self.g(t_a0, self.g_func)
            
            # save g(z) = a
            self.A.append(np.copy(t_a0))

        # calculate Wa + B
        t_a0 = np.matmul(self.weights[-1], t_a0) + self.biases[-1]
        self.Z.append(np.copy(t_a0))
        
        # apply activation
        t_a0 = self.g(t_a0, self.h_func)
        
        # save g(z) = a
        self.A.append(np.copy(t_a0))
        
    def calculate_α(self):

        '''
        INPUTS
        A = list of activations of each layer
            0th element = (m,d,1) numpy array of activations of input layer, 
            1st element = (m,d,1) numpy array of activations of first hidden layer, etc
        Z = list of values Wa + B for each layer
            0th element = (m,d,1) numpy array of W_1a_0 + B_1 for a0 the (m,d,1) array of inputs, 
            1st element = (m,d1,1) numpy array of W_2a_1 + B_2 for a1 the (m,d1,1) array of first hidden layer activations, etc
        W = list of numpy arrays with weights for each layer
        B = list of numpy arrays with biases for each layer
        g_func = activation function tag for hidden layers
        dg = master function with all activation function derivatives

        SETS VALUES OF
        α = list of matrices (α_m)_ijk = (W_m)_jk * dg(q_ik) for q = z^(m-1)
            0th element = W_2 * dg(z_1), 1st element = W_3 * dg(z_2), etc
        '''

        self.α = []

        # loop over layers, match ith element of Z with (i+1)th element of W
        for idx in range(len(self.weights)-1):

            # size of current layer
            n = np.shape(self.weights[idx+1])[0]

            # tile z since regular broadcasting doesn't work with current array sizes & transpose each slice (i,:,:)
            zT_tile = np.transpose(np.tile(self.Z[idx], (1, 1, n)), (0,2,1))

            # save a copy to α
            self.α.append(np.copy(self.weights[idx+1] * self.dg(zT_tile, self.g_func)))

    def calculate_Λ(self):

        '''
        INPUTS
        α = list of matrices (α_m)_ijk = (W_m)_jk * dg(q_ik) for q = z^(m-1)
            0th element = W_2 * dg(z_1), 1st element = W_3 * dg(z_2), etc

        SETS VALUES OF
        Λ = list of numpy arrays Λ_(n-m) = α_n α_(n-1) ... α_(n-m+1)
            0th element = Λ_1, 1st element = Λ_2, etc
            i.e. 1st element = associated with gradients of 1st layer weights, 2nd element with 2nd layer weights, etc
        '''

        # create (m,d,d) array with each slice the (d,d) identity for d the size of the output to initialize
        d = np.shape(self.α[-1])[1]
        m = np.shape(self.α[-1])[0]
        self.Λ = [np.tile(np.identity(d), (m, 1, 1))]

        # successively build up products to minimize number of matrix products-- have to do in reverse order of αs
        for idx in range(len(self.α)):

            self.Λ.append(np.matmul(self.Λ[idx], self.α[len(self.α) - 1 - idx]))

        # reverse to match ordering convention
        self.Λ.reverse()
        
        # if using softmax output activation, calculate Γ also
        if self.h_func == 'soft_max':
            
            '''
            # grab values to work with
            t_z = self.Z[-1]
            t_af = self.A[-1]
            
            # prepare exp(z_iℓ - z_ij)
            zT = np.transpose(t_z, (0, 2, 1))
            exp_δz = np.exp(t_z - zT)
            '''
            
            # values to work with
            t_af = self.A[-1]
            
            # calculate adjustment and subtract from Λ
            for idx in range(len(self.Λ) - 1):
                
                # a_ijΣ = np.matmul(np.transpose(exp_δz, (0, 2, 1)), self.Λ[idx]) * t_af
                Σ = np.matmul(np.transpose(self.Λ[idx], (0, 2, 1)), t_af)
                self.Λ[idx] = self.Λ[idx] - np.transpose(Σ, (0, 2, 1))  

    def calculate_η(self):

        '''
        INPUTS
        A = list of activations of each layer
            0th element = (m,d,1) numpy array of activations of input layer, 
            1st element = (m,d,1) numpy array of activations of first hidden layer, etc
        Z = list of values Wa + B for each layer
            0th element = (m,d,1) numpy array of W_1a_0 + B_1 for a0 the (m,d,1) array of inputs, 
            1st element = (m,d1,1) numpy array of W_2a_1 + B_2 for a1 the (m,d1,1) array of first hidden layer activations, etc
        L_func = loss function tag
        h_func = activation function tag for output layer
        dL = master function with derivatives of loss functions
        dg = master function with all activation function derivatives
        y = expected outputs (for cost/loss function), must be (m,d,1) numpy array for d outputs with m examples

        SETS VALUES OF
        (m,d,1) numpy array of dL(a_n) * dh(z_n) (element-wise)
        '''

        self.η = self.dL(self.A[-1], self.y, self.L_func) * self.dg(self.Z[-1], self.h_func)

    def calculate_derivatives(self):

        '''
        INPUTS
        A = list of activations of each layer
            0th element = (m,d,1) numpy array of activations of input layer, 
            1st element = (m,d,1) numpy array of activations of first hidden layer, etc
        Λ = list of numpy arrays Λ_(n-m) = α_n α_(n-1) ... α_(n-m+1)
            0th element = Λ_1, 1st element = Λ_2, etc
            i.e. 1st element = associated with gradients of 1st layer weights, 2nd element with 2nd layer weights, etc
        η = (m,d,1) numpy array of dL(a_n) * dh(z_n) (element-wise)

        SETS VALUES OF
        dJdW = list of numpy arrays with gradients relative to weights of each layer
            dJdW[0] = numpy array w/ (ij)th element = gradient of cost J relative to (ij)th element of 1st layer weights
        dJdB = list of numpy arrays with gradients relative to biases of each layer
            dJdB[0] = numpy array w/ (i)th element = gradient of cost J relative to (i)th element of 1st layer biases
        '''

        self.dJdW = []
        self.dJdB = []
        for idx in range(len(self.Λ)):

            # create matrix Δ
            Δ = np.squeeze(np.matmul(np.transpose(self.Λ[idx], (0,2,1)), self.η)) # kill singlet dimension

            # calculate matrix dJdW
            self.dJdW.append(np.matmul(np.transpose(Δ), np.squeeze(self.A[idx]))) # kill singlet dimension in a_iν also

            # calculate vector dJdB (have to add singlet dimension for later)
            self.dJdB.append(np.expand_dims(np.sum(Δ, axis = 0), axis = 1))

    def calculate_gradJ(self):

        '''
        INPUTS
        a0 = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m training examples
        W = list of numpy arrays with weights for each layer
        B = list of numpy arrays with biases for each layer
        g = master function with all activation functions
        dg = master function with all activation function derivatives
        dL = master function with derivatives of loss functions
        g_func = activation function tag for hidden layers
        h_func = activation function tag for output layer
        L_func = loss function tag
        y = expected outputs (for cost/loss function), must be (m,d,1) numpy array for d outputs with m examples

        SETS VALUES OF
        dJdW = list of numpy arrays with gradients relative to weights of each layer
            dJdW[0] = numpy array w/ (ij)th element = gradient of cost J relative to (ij)th element of 1st layer weights
        dJdB = list of numpy arrays with gradients relative to biases of each layer
            dJdB[0] = numpy array w/ (i)th element = gradient of cost J relative to (i)th element of 1st layer biases
        '''

        # calculate layer activations
        self.calculate_layer_activations()

        # calculate alphas
        self.calculate_α()

        # calculate Λ products
        self.calculate_Λ()

        # calculate η
        self.calculate_η()

        # calculate gradients
        self.calculate_derivatives()
        
    '''
    ACTIVATION AND LOSS FUNCTIONS (for use within methods, not for external calls)
    g = activations
    dg = derivatives of activations
    L = loss function
    dL = derivatives of loss function
    J = cost function
    '''
        
    def g(self, z, func):

        # linear
        if func == 'linear':
            return z

        # sigmod
        elif func == 'sigmoid':
            return 1/(1 + np.exp(-z))

        # reLU
        elif func == 'reLU':
            return np.where(z > 0.0, z, 0.0)
        
        # softmax
        elif func == 'soft_max':
            
            # calculate max z value to shift everything
            z_max = np.reshape(np.amax(z, axis = 1), (np.shape(z)[0], 1, 1))
            
            # calculate exponents
            exp_z = np.exp(z - z_max)
            
            # calculate slice sums (have to do some reshaping for broadcasting
            sum_exp_z = np.reshape(np.sum(exp_z, axis = 1), (np.shape(exp_z)[0], 1, 1))
            
            # combine
            return exp_z / sum_exp_z

        # default to reLU for all other inputs
        else:
            return np.where(z > 0.0, z, 0.0)

    def dg(self, z, func):

        # linear
        if func == 'linear':
            return np.ones(np.shape(z))

        # sigmoid
        elif func == 'sigmoid':
            return np.exp(-z)/np.square(1 + np.exp(-z))

        # reLU
        elif func == 'reLU':
            return np.where(z > 0.0, 1.0, 0.0)
        
        # softmax
        elif func == 'soft_max':
            
            '''
            # calculate max z value to shift everything
            z_max = np.reshape(np.amax(z, axis = 1), (np.shape(z)[0], 1, 1))
            
            # calculate exponents
            exp_z = np.exp(z - z_max)
            
            # calculate slice sums (have to do some reshaping for broadcasting
            sum_exp_z = np.reshape(np.sum(exp_z, axis = 1), (np.shape(exp_z)[0], 1, 1))
            
            # combine to get original p dist
            P = exp_z / sum_exp_z
            
            # and return P - P^2
            return P - np.square(P)
            '''
            
            # calculate layers
            self.calculate_layer_activations()
            
            # not the actual softmax derivative, but acts structurally as it
            return self.A[-1]
        
        # default to reLU for all other inputs
        else:
            return np.where(z > 0.0, 1.0, 0.0)

    def J(self, a, y, func):

        # mean square error
        if func == 'mean_square_error':
            return (1/(2 * np.shape(a)[0])) * np.sum(np.square(a - y))
        
        # soft max
        elif func == 'cross_entropy':
            
            # adjust a to filter really small values
            # t_a = np.where(a > 10**-20, a, 10**-20)
            
            # filter to make sure rounding doesn't mess with us
            t_L = np.where(y > 0.5, -y * np.log(a), 0.0)
            
            # do the sum
            return np.sum(t_L) / np.shape(a)[0]

        # default to mse
        else:
            return (1/(2 * np.shape(a)[0])) * np.sum(np.square(a - y))

    def L(self, a, y, func):

        # mean square error
        if func == 'mean_square_error':
            return (1/(2 * np.shape(a)[0])) * np.square(a - y)
        
        # soft max
        elif func == 'cross_entropy':
            
            # adjust a to filter really small values
            # t_a = np.where(a > 10**-20, a, 10**-20)
            
            # filter to make sure rounding doesn't mess with us
            t_L = np.where(y > 0.5, -y * np.log(a), 0.0)
            
            return t_L / np.shape(a)[0]

        # default to mse
        else:
            return (1/(2 * np.shape(a)[0])) * np.square(a - y)

    def dL(self, a, y, func):

        # mean square error
        if func == 'mean_square_error':
            return (1/(np.shape(a)[0])) * (a - y)
        
        # soft max
        elif func == 'cross_entropy':
            
            # adjust a to filter really small values
            # t_a = np.where(a > 10**-20, a, 10**-20)
            
            # filter to make sure rounding doesn't mess with us
            t_L = np.where(y > 0.5, -y / a, 0.0)
            
            return t_L / np.shape(a)[0]

        # default to mse
        else:
            return (1/np.shape(a)[0]) * (a - y)
        
        
    '''
    FUNCTIONS FOR ASSIGNMENT AND RETRIEVAL BY USER
    predict = predict outputs for given set of inputs
    '''
    
    # forward propagation
    def predict(self, a0):

        '''
        INPUTS
        a0 = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m training examples
        weights = list of numpy arrays with weights for each layer
        biases = list of numpy arrays with biases for each layer
        g_func = activation function tag for hidden layers
        h_func = activation function tag for output layer
        g = master function with all activation functions
        
        SETS VALUES OF
        af = numpy array of predictions, must be (m,d,1) numpy array for output layer of d neurons with m training examples
        '''
        
        # temporary a0 to avoid overwriting
        t_a0 = np.copy(a0)
        
        # propagate network
        for idx in range(len(self.weights)-1):

            # multiply to advance by one layer
            t_a0 = self.g(np.matmul(self.weights[idx], t_a0) + self.biases[idx], self.g_func)
        
        return self.g(np.matmul(self.weights[-1], t_a0) + self.biases[-1], self.h_func)
        
    def load_data_set(self, train_input, train_output, test_input):
        
        '''
        INPUTS
        train_input = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m training examples
        train_output = expected outputs, must be (m,d,1) numpy array for d outputs with m examples
        test_input = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m testing examples
        
        SETS VALUES OF
        train_input = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m training examples
        train_output = expected outputs, must be (m,d,1) numpy array for d outputs with m examples
        test_input = numpy array of inputs, must be (m,d,1) numpy array for input layer of d neurons with m testing examples
        '''
        
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        
    def calculate_cost(self):
        
        '''
        INPUTS
        a0 = numpy array of inputs, must be (m,d,1) numpy array for output layer of d neurons with m training examples
        y = expected outputs, must be (m,d,1) numpy array for d outputs with m examples
        J_func = form of cost function
        
        OUTPUTS
        Current cost
        '''
        
        t_af = self.predict(self.a0)
        return self.J(t_af, self.y, self.J_func)
        
    '''
    GRADIENT DESCENT
    '''
    
    def grad_descent_step(self):
        
        # calculate gradient
        self.calculate_gradJ()
        
        # go through the layers and adjust the weights and biases
        for idx in range(len(self.weights)):
            
            self.weights[idx] = self.weights[idx] - (self.l_rate * self.dJdW[idx])
            self.biases[idx] = self.biases[idx] - (self.l_rate * self.dJdB[idx])
            
    def train_network(self, n_epochs, batch_size, learn_rate):
        
        # assign the learning rate
        self.l_rate = learn_rate
        
        for idx in range(n_epochs):
            
            # select a random sample of the data to train on for this epoch
            batch_idx = random.sample(range(0, np.shape(self.train_input)[0]), batch_size)
            
            # assign a0, y based on random sample
            self.a0 = np.copy(self.train_input[batch_idx,:])
            self.y = np.copy(self.train_output[batch_idx,:])
            
            # do a step of gradient descent
            self.grad_descent_step()
            
            # display string with information eg cost, epoch #, etc
            t_cost = self.J(self.A[-1], self.y, self.J_func)
            
            # break if nan
            if np.isnan(t_cost):
                print('we borked it')
                break
            
            # print info
            print(f'Epoch {idx} ====== Current Cost: {t_cost:.5f}')