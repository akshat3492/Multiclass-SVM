"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        
        num_train, dimension = X_train.shape
        grads = self.w * self.reg_const #(D,C)
        batch_score = X_train.dot(self.w)#(N,C)
        true_score = batch_score[np.arange(num_train), y_train]#(N,)
        true_score = true_score.reshape(-1,1)#(N,1)        
        grad_score = ((true_score-batch_score)<1).astype(int)#(N,C)
        #print(grad_score.shape)
        grad_score[np.arange(num_train), y_train] = 0#(N,C)
        true_grad = grad_score.sum(axis=1).reshape(-1,1)*X_train#(N,D)

        for i in range(num_train):
            grads[:,y_train[i]] -= true_grad[i] #check (D,C) and (N,D)
            grads += X_train[i].reshape(-1,1) * grad_score[i]
            
        #-------------------------------------or-------------------------------------
        #Source - https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/
        num_train, dimension = X_train.shape
        grads = self.w * self.reg_const #(D,C)
        batch_score = X_train.dot(self.w)#(N,C)
        true_score = batch_score[np.arange(num_train), y_train]#(N,)
        grad_score = np.maximum(0, batch_score - true_score[:,np.newaxis] + 1)
        grad_score[np.arange(num_train), y_train] = 0
        X_mask = np.zeros(grad_score.shape)
        X_mask[grad_score>0] = 1
        true_grad = X_mask.sum(axis=1).reshape(-1,1)*X_train
        for i in range(num_train):
            grads[:,y_train[i]] -= true_grad[i] #check (D,C) and (N,D)
            grads += X_train[i].reshape(-1,1) * X_mask[i]
        
        return grads

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        n_data, dimension = X_train.shape
        batch_size = 100
        
        self.w = np.random.rand(dimension, self.n_class)#(D, C)
        for i in range(self.epochs):
            indices = np.random.choice(n_data, batch_size)
            X_batch = X_train[indices] #(batch_size, D)
            y_batch = y_train[indices]#(batch_size, 1)
            self.w = self.w - self.lr * self.calc_gradient(X_batch, y_batch)  
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pred = X_test.dot(self.w)
        return pred.argmax(axis=1)