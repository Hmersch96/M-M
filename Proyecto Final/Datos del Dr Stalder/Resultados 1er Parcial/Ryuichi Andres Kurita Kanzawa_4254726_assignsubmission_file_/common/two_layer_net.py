import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    #input_size = cantidad de neuronas de entrada (X)
    #hidden_size = cantidad de neuronas de 1er layer (oculto)
    #output_size = cantidad de neuronas de salida
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params={}
        self.params['W1']=weight_init_std * np.random.randn(input_size,hidden_size) #W1 = 6x3       b1 = 3 W2 = 3x2  b2 = 2 
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1']) 
        self.layers['Relu1']=ReLu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x

    def ResultadoObtener(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = softmax(a2)
        return y

    def devolverParametros(self):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        return W1,W2,b1,b2

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        if t.ndim != 1 : t=np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W= lambda W: self.loss(x,t)
        grads={}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def setParameters(self , W1, W2, b1, b2):
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2