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
        self.X=X
        self.Y=Y

        # Implementing gradient descent
        for i in range(self.epochs):
            self.update_weights()
    def update_weights(self):
        Y_cap = 1 / (1+np.exp(-(self.X.dot(self.w)+self.b))) #wx+b

        # derivative
        dw = (1/self.m)*np.dot(self.X.T,(Y_cap - self.Y))
        db = (1/self.m)*np.sum(Y_cap -self.Y)

        # updating the weights using gradient descent
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self):
        Y_pred = 1 / (1+np.exp(-(self.X.dot(self.w)+self.b)))
        Y_pred = np.where(Y_pred>0.5,1,0)
        return Y_pred