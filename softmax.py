"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        for i in range(len(X_train)):
            X_temp = X_train[i]#(1,D)
            y_temp = y_train[i]#scalar
            
            true_score = np.dot(self.w[y_temp], X_temp.T) #(1,D) * (D,1) -> (1,1)
            #predict_score = np.dot(self.w, X_temp.T)#(C,1)
            #predict_temp_index = np.argmax(predict_score)
            for j in range(self.n_class):
                predict_score = np.dot(self.w[j], X_temp.T)#(1,1)
                if true_score - predict_score < 1:
                    self.w[y_temp] = self.w[y_temp] + self.lr*X_temp
                    self.w[j] = self.w[j] - self.lr * X_temp
                self.w[j] = self.w[j] - (self.w[j]*self.lr*self.reg_const)/len(X_train)
     
        return self.w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        n_data, dimension = X_train.shape
        #n_class = self.class
        batch_size = 100
        
        self.w = np.random.rand(self.n_class, dimension)#(C, D)
        #print(self.w)
        for i in range(self.epochs):
            #print(i)
            indices = np.random.choice(n_data, batch_size)
            X_batch = X_train[indices] #(batch_size, D)
            y_batch = y_train[indices]#(batch_size, 1)
            
            self.w = self.calc_gradient(X_batch, y_batch)
        #print(self.w)
        #return grad_w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        preds = np.argmax(np.dot(self.w, X_test.T), axis=0)#(C,D) * (D,N) -> (C,N)
        #print(preds)
        return preds
