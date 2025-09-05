import numpy as np
class logistic_regression:
    # declaring the learning rate and the number of iterations (hyperparameters)
    def __init__(self,learning_rate,epochs):
        self.learning_rate=learning_rate
        self.epochs=epochs
    
    # fit function to train the model with some dataset
    def fit(self,X,Y):
        # number of datapoints in the dataset ----> m
        # Number of input features in the dataset ----> n
        self.m,self.n = X.shape

        # Initiating the weight value and bias value
        self.w=np.zeros(self.n)
        self.b=0
        self.x=X
        self.y=Y

        # Implementing gradient descent
        for i in range(epochs):
            self.update_weights()
    def update_weights(self):
        pass
    def predict(self):
        pass