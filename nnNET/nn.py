import h5py
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases import backward_propagation_test_case, predict_test_case, update_parameters_test_case, \
    compute_cost_test_case, nn_model_test_case, initialize_parameters_test_case, forward_propagation_test_case

# 设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。
#np.random.seed(1)
X, Y = load_planar_dataset()

# 绘制散点图
plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0,:].shape), s=40, cmap=plt.cm.Spectral)
#plt.show()

#X, Y = X.T, Y.T
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
#X= 400 个点的x坐标，y坐标
#Y = 对应index点的属性

def fit_by_logic(X,Y):

    clf = sklearn.linear_model.LogisticRegressionCV()
    #用逻辑回归进行拟合 仅 wx+b ？
    clf.fit(X.T, Y.T)
    print("Coefficients: ", clf.coef_)


    # Plot the decision boundary for logistic regression
    # 绘制决策边界
    plot_decision_boundary(lambda a: clf.predict(a), X, Y)
    """plot_decision_boundary在绘制时就可以直接调用这个lambda函数对各个坐标进行预测,而不需要用户每次定义新变量。"""
    """ 其实也就是把图标上所有点都输入进去了"""


    # 图标题
    plt.title("Logistic Regression")
    plt.show()


    # 打印准确性
    LR_predictions = clf.predict(X.T)
    # perdict的结果
    print ('逻辑回归的准确性：%d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
           '% ' + "(正确标记的数据点所占的百分比)")

#---------------------------------构建 单隐层网----------------------------------------

# 设置层数
def layer_size(n_h,X,Y):

    M = X.shape[1] #标签数量
    N = X.shape[0] #分类种类
    y_size = Y.shape[0]
    x_size = N

    h_size = n_h
    """h_size 隐层神经元个数"""

    return x_size, y_size, h_size


#初始化参数
def init_params(lay_sizes):

    x_size, y_size, h_size = lay_sizes

    np.random.seed(2)
    W1 = np.random.randn(h_size, x_size) * 0.01
    b1 = np.zeros((h_size, 1))
    W2 = np.random.randn(y_size, h_size) * 0.01
    b2 = np.zeros((y_size, 1))
    # w矩阵设置的维数，即为了能够进行矩阵乘  根据矩阵点乘 （h,x）dot (x,1) = (h,1)

    #print("W1 = " + str(W1.shape), "W2 = " + str(W2.shape), "b1 = " + str(b1.shape), "b2 = " + str(b2.shape))
    # 使用断言确保我的数据格式是正确的
    assert (W1.shape == (h_size, x_size))
    assert (b1.shape == (h_size, 1))
    assert (W2.shape == (y_size, h_size))
    assert (b2.shape == (y_size, 1))

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return params

#测试一下 initialize_parameters 函数
print("=========================测试initialize_parameters=========================")
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = init_params((n_x,n_y,n_h))
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# 测试初始化函数
#res_dic = init_params(layer_size())
#print("w1: ",res_dic['w1'], "\nb1: ", res_dic['b1'], "\nw2: ", res_dic['w2'], "\nb2: ", res_dic['b2'])


#relu 函数
def rectified(x):

  return max(0.0, x)

#前向传播
def propagation(w1, w2, b1, b2, X):

    # print(w1.shape, X.shape)

    Z1 = np.dot(w1, X) + b1

    A1 = np.tanh(Z1)

    #print(w2.shape, A1.shape)
    #A1 = rectified(Z1)

    Z2 = np.dot(w2, A1) + b2

    A2 = sigmoid(Z2)
    """直接两段前向传播完"""
    """隐藏层用relu，输出二分用sig"""

    assert A2.shape == (1, X.shape[1]), "Z1 h as the wrong shape"

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return A2,cache

# 测试一下 forward_propagation 函数
print("=========================测试forward_propagation=========================")
X_assess, parameters = forward_propagation_test_case()
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
A2, cache = propagation(W1,W2,b1,b2,X_assess)
# 我们在这里使用均值只是为了确保你的输出与我们的输出匹配。
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


def compute_cost(A2, Y):
    """
    计算方程（7）中给出的交叉熵成本，

    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         Y - "True"标签向量,维度为（1，数量）
         parameters - 一个包含W1，B1，W2和B2的字典类型的变量

    返回：
         成本 - 交叉熵成本给出方程（7）
    """

    # 样本数量
    m = Y.shape[1]

    # 计算交叉熵代价
    logprobs = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = -1 / m * np.sum(logprobs)

    # 确保损失是我们期望的维度
    # 例如，turns [[17]] into 17
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost

# 测试一下 compute_cost 函数
print("=========================测试compute_cost=========================")
A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess)))


#  反向传播
def backward_pro(X,Y,cache,params):

    m = X.shape[1]

    W1 = params['W1']
    W2 = params['W2']

    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    dZ2 = A2 - Y
    #交叉熵一阶导

    dW2 = 1/m * np.dot(dZ2, A1.T)
    dB2 = 1/m * np.sum(dZ2, axis = 1, keepdims=True)


    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    #cost = loss( sig(Z2) )
    # dl/dZ1 = f'(loss)dz1  交叉熵方程求导

    dW1 = 1 / m * np.dot(dZ1, X.T)
    # dw1 = dL/dA * dA/dy *dy/dW1
    #dA = dZ, dy/dw1 近似为x

    dB1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    # dl/db = dl/dA * dA/dy * dy/dx   dy/dx =>> 1

    back_res = {'dW1': dW1,
                'dW2': dW2,
                'db1': dB1,
                'db2': dB2}

    return back_res

# 测试一下 backward_propagation 函数
print("=========================测试backward_propagation=========================")
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_pro(X_assess, Y_assess,cache,parameters)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


# 更新参数
def optimizer(params, grads, learning_rate):

    W1 = params['W1']
    B1 = params['b1']
    W2 = params['W2']
    B2 = params['b2']

    dW1 = grads['dW1']
    dB1 = grads['db1']
    dW2 = grads['dW2']
    dB2 = grads['db2']

    # 每个参数的更新规则
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2

    parameters = {"W1": W1,
                  "b1": B1,
                  "W2": W2,
                  "b2": B2}

    return parameters

# 测试一下 update_parameters 函数
print("=========================测试update_parameters=========================")
parameters, grads = update_parameters_test_case()
parameters = optimizer(parameters, grads,learning_rate=1.2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def nn_model1(X, Y, n_h, num_iterations=10000,lr = 1.2, print_cost=False):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
     """

    # 初始化参数，然后检索 W1, b1, W2, b2。输入:“n_x, n_h, n_y”。
    np.random.seed(3)
    n_x, n_y, n_h = layer_size(n_h, X, Y)

    # 初始化参数，然后检索 W1, b1, W2, b2。
    # 输入:“n_x, n_h, n_y”。输出=“W1, b1, W2, b2，参数”。
    parameters = init_params((n_x, n_y, n_h))
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 循环(梯度下降)
    for i in range(0, num_iterations):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # 前项传播
        A2, cache = propagation(W1,W2, b1, b2, X)

        # 计算成本
        cost = compute_cost(A2, Y)

        # 反向传播
        grads = backward_pro(X, Y,cache,parameters)

        # 更新参数
        parameters = optimizer(parameters, grads, lr)

        # 每1000次迭代打印成本
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

#整合
def nn_model(X, Y, h_size, num_iterations=10000, lr = 1.2 ,print_cost=False):

    #设置layer参数
    lay_size = layer_size(h_size,X,Y)

    #初始化参数
    parameters = init_params(lay_size)



    for i in range(0, num_iterations):

        w1 = parameters["W1"]
        w2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        A2,caches = propagation(w1,w2, b1, b2, X)
        # 前向传播

        cost = compute_cost(A2, Y)

        gra = backward_pro(X, Y, caches, parameters)

        parameters = optimizer(parameters, gra, lr)
        #更新参数

        # 每1000次迭代打印成本
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

        #print(parameters['W1'])

    return parameters
# 测试一下 nn_model 函数
print("=========================测试nn_model=========================")
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))




def predict(parameters, X):
    """
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

     """

    # 使用前向传播计算概率，并使用 0.5 作为阈值将其分类为 0/1。
    w1 = parameters["W1"]
    w2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    A2, cache = propagation(w1,w2,b1,b2,X)
    predictions = np.round(A2)

    return predictions

"""
# 测试一下 predict 函数
print("=========================测试predict=========================")
parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("预测的平均值= " + str(np.mean(predictions)))
"""


h_size = 20
tmp = nn_model(X, Y, h_size, num_iterations=10000, lr = 1.2,print_cost=True)


# 绘制决策边界
plot_decision_boundary(lambda x: predict(tmp, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(h_size))
plt.show()


