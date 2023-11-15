import numpy as np
import h5py
import matplotlib.pyplot as plt
#from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
#import lr_utils


#初始化参数
def initialize_parameters_deep(layer_dims):

    np.random.seed(1)
    parameters = {}
    for i in range(len(layer_dims)-1):
        parameters[f'W{str(i+1)}'] = np.random.randn(layer_dims[i+1],layer_dims[i])/np.sqrt(layer_dims[i])
        parameters[f'b{str(i+1)}'] = np.random.randn(layer_dims[i+1], 1)

        # 确保我要的数据的格式是正确的
        assert (parameters['W' + str(i+1)].shape == (layer_dims[i+1], layer_dims[i])), "init W wrong"
        assert (parameters['b' + str(i+1)].shape == (layer_dims[i+1], 1)), "init B wrong"

    return parameters




#前向传播+激活函数
def linear_activation_forward(A_prev, W, b, activation):

    A = np.zeros(A_prev.shape)
    cache = (1,2)

    Z = np.dot(W, A_prev) + b
    Lin_cache = (A_prev,W,b)

    if activation == 'relu':
        A, cache = relu(Z)

    if activation == 'sigmoid':
        A, cache = sigmoid(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1]), 'A_dim error'
    #因为数据信息不能丢，于是A[1]永远是数据信息

    cache_res = (Lin_cache,cache)

    return A, cache_res



#收集所有前向传播的cache
#创建cache字典
def L_model_forward(X, parameters):

    A = X
    caches = []
    L = len(parameters)//2
    for i in range(1,L):

        A, cache = linear_activation_forward(A, parameters[f"W{str(i)}"],parameters[f"b{str(i)}"],"relu")
        #cache = ([Ai-1,Wi,bi],[Ai])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters[f"W{L}"],parameters[f"b{L}"],"sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

#计算cost
def compute_cost(AL, Y):

    m = Y.shape[1]
    #样本数量

    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1, keepdims=True)

    cost = np.squeeze(cost)
    #squeeze:消除无用维度

    return cost

#带有L2正则化的cost
def compute_cost_with_regularization(AL, Y, caches, lambd=0.7):

    m = Y.shape[1]
    # 样本数量

    L = len(caches)

    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1, keepdims=True)

    L2_regularization_cost = 1

    for i in range(L):
        Lin_cache = caches[L - i - 1]
        W = Lin_cache[1]
        L2_regularization_cost += np.sum(np.square(W))

    L2_regularization_cost *= lambd/2/m

    cost += L2_regularization_cost

    cost = np.squeeze(cost)
    # squeeze:消除无用维度

    return cost


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]
    #样本数量

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dW.shape == W.shape, "dW.dim error"
    assert db.shape == b.shape, "b.dim error"
    assert (dA_prev.shape == A_prev.shape), "dA.dim error"

    return dA_prev, dW, db



#反向传播函数
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    dA_prev, dW, db = 0,0,0

    if activation =='relu':

        dZ = relu_backward(dA , activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)


    if activation == 'sigmoid':

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db




#计算反向传播
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)


    # 初始化反向传播

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #交叉熵的导数

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for i in range(L - 1):
        current_cache = caches[L-2-i]
        grads["dA" + str(L-1-i)], grads["dW" + str(L-i-1)], grads["db" + str(L-1-i)] = linear_activation_backward(grads[f'dA{str(L-i)}'], current_cache, 'relu')

    return grads

#正则化反向传播
#反向传播函数
def linear_activation_backwardWithRGE(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache

    W = linear_cache[1]
    A = linear_cache[0]

    m = A.shape[1]

    dA_prev, dW, db = 0,0,0

    if activation =='relu':

        dZ = relu_backward(dA, activation_cache)
        #dZ = np.multiply(dA2,np.int64(A2 > 0))  对relu求导的具体

        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        dW +=  lambd * W / m


    if activation == 'sigmoid':

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        dW += lambd * W / m

    return dA_prev, dW, db



#带有正则化打反向传播
def backward_propagation_with_regularization(AL, Y, caches, lambd):

    grads = {}
    m = AL.shape[1]
    L = len(caches)
    current_cache = caches[L-1]

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backwardWithRGE(dAL, current_cache,
                                                                                                  'sigmoid',lambd)

    for i in range(L - 1):
        current_cache = caches[L - 2 - i]
        grads["dA" + str(L - 1 - i)], grads["dW" + str(L - i - 1)], grads[
            "db" + str(L - 1 - i)] = linear_activation_backwardWithRGE(grads[f'dA{str(L - i)}'], current_cache, 'relu', lambd)

    return grads


#修改参数
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters)//2

    for i in range(L):
        W_tmp = parameters[f'W{str(i+1)}']
        b_tmp = parameters[f"b{str(i+1)}"]

        dW, db = grads[f"dW{i+1}"], grads[f"db{i+1}"]
        parameters[f'W{str(i + 1)}'] = W_tmp - dW * learning_rate
        parameters[f'b{str(i + 1)}'] = b_tmp - db * learning_rate


    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False, isPlot=True, isReg = False):

    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):

        AL, caches = L_model_forward(X, parameters)

        if isReg == False:
            cost = compute_cost(AL,Y)
            costs.append(cost)
            grads = L_model_backward(AL, Y, caches)

        else:

            cost =  compute_cost_with_regularization(AL, Y, caches, lambd=0.7)
            grads = backward_propagation_with_regularization(AL, Y, caches, lambd=0.7)

            costs.append(cost)

        if (print_cost == True) and i % 100 == 0:
            print(f"-----------epoch = {i}--------\nLoss = {cost}")

        parameters = update_parameters(parameters,grads,learning_rate)

    # 迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('Loss')
        plt.xlabel('iterations (per tens)')
        plt.title(f"Learning rate ={(learning_rate)},layer_size = {len(layers_dims)}")
        plt.show()

    return parameters


def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print(f"准确度为: { str(float(np.sum((p == y)) / m*100))}%")

    return p

