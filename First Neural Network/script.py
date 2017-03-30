from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time
        random.seed(1234)
        
        # 2 layers consisting each of 3 neurons
        self.synaptic_weights = [self.layer_weights(3),
                                 self.layer_weights(3)]
  
    # Gives weights for a layer consisting of n nodes.
    def layer_weights(self, n):
        # Initialize empty array
        layer_weights = []
        
        # Create one set of weights for eachnode
        for num in range(n):
            layer_weights.append(self.neuron_weights())
            
        return layer_weights
      
    # Gives 3x1 matrix with values from -1 to 1 and mean 0. To be used as
    # weights for one neuron.
    def neuron_weights(self):
        return 2 * random.random((3,1)) - 1
    
    # The sigmoid function, which describes an S-shaped curve.
    # The weighted sum of the inputs is passed through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1-x)
    
    def predict(self, inputs):        
        # Pass input through our neural network. First go through each layer:
        for layer in self.synaptic_weights:
            # Get the outputs from each of the neurons
            out = array([self.__sigmoid(dot(inputs,layer[0])),
                         self.__sigmoid(dot(inputs,layer[1])),
                         self.__sigmoid(dot(inputs,layer[2]))])
            
            # Redirect them to the next layer
            inputs = out.T
            
        # The last "inputs" are summed (or given through a neuron with all
        # weights = 1) and put through sigmoid
        return self.__sigmoid(dot(inputs,[1,1,1]))
            
    ###### Is still made for a 1 layer 1 neuron network. Doesn't work
    def train(self, training_set_inputs, training_set_outputs, 
              number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural net
            output = self.predict(training_set_inputs)
            
            # Calculate the error
            error = training_set_outputs - output
            
            # Multiply the error by the input and again by the gradient of the 
            # Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the 
            # weights.
            adjustment = dot(training_set_inputs.T,
                             error * self.__sigmoid_derivative(output))
            
            # Adjust the weights
            self.synaptic_weights += adjustment

if __name__ == "__main__":

    # Initalises a single neuron neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights:')
    print(neural_network.synaptic_weights)

    # Training set: We have 4 examples, each containing 3 inputs
    #               and 1 output
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set
    # Do it 10,000 times and make small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weights after training:')
    print(neural_network.synaptic_weights)

    # Test the neural network
    print('Predicting:')
    print(neural_network.predict(array([1, 0, 0])))
