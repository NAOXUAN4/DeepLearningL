import  matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import scipy.misc
import cv2


#-----------------------------------------预处理--------------------------------------------------------

#1.找出数据的尺寸和维度（m_train，m_test，num_px等）
#2.重塑数据集，以使每个示例都是大小为（num_px * num_px * 3，1）的向量
#3.“标准化”数据

"""1.读取数据集"""
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
# 打印出当前的训练标签值
# 使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
# print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
# 只有压缩后的值才能进行解码操作
"""
index = 0
cv2.imshow('img',train_set_x_orig[index])
cv2.waitKey(0)
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
"""

m_train = train_set_x_orig.shape[0]   # 训练集里图片的数量。
m_test = test_set_x_orig.shape[0]     # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]    # 训练集里图片的宽度
num_py = train_set_x_orig.shape[2]    # 训练集里图片的宽度

# #看一看 加载的东西的具体情况
#print ("Number of training examples: m_train = " + str(m_train))
#print ("Number of testing examples: m_test = " + str(m_test))
#print ("Height of each image: num_px = " + str(num_px))
#print ("Each image is of size: (" + str(num_px) + ", " + str(num_py) + ", 3)")
#print ("train_set_x shape: " + str(train_set_x_orig.shape))
# test_set_y_orig 为局部变量，返回赋给 train_set_y 了
#print ("train_set_y shape: " + str(train_set_y.shape))
#print ("test_set_x shape: " + str(test_set_x_orig.shape))
#print ("test_set_y shape: " + str(test_set_y.shape))

"""单张训练数据（64，64，3）"""
"""训练集数据为（209张 ， 64， 64，3）"""
"""单张测试数据（64，64，3）"""
"""测试集数据为（50张 ， 64， 64，3）"""


"""2.打平图片维度为（x*y*3,1）；为了一张图片的所有信息，作为一个维度，全部输入神经网络"""
# X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
# 将训练集的维度降低并转置。
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
"""这样直观来说就是图像每一个像素点作为高维，即这个这个像素点有几十个从属，每个从属都是不同一张图片的同一个位置的像素点的颜色信息"""
#print(len(train_set_x_flatten[0]))
#print(len(train_set_x_flatten))
"""
[255,255,255.....*209个]
[255,255,255.....*209个]
......
......
64*64*3 个
(12288行, 209列)
"""


# 看看降维之后的情况是怎么样的
#print ("训练集降维最后的维度: " + str(train_set_x_flatten.shape))
#print ("训练集_标签的维数: " + str(train_set_y.shape))
#print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
#print ("测试集_标签的维数: " + str(test_set_y.shape))


"""3.数据集的存图片数据的维度（12288），每一位都存着 0~255 的数字."""
"""即表示像素点 R/G/B的值，而255现在把他们打成0~1的float，方便训练"""

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#-----------------------------------------构建神经网络---------------------------------------------------

#1.sigmoid函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#测试sigmoid函数
"""
print(sigmoid(-10))
print(sigmoid(10))
可知sigmoid函数（0：1）
"""

#2.初始化参数函数
def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    """系数w，第一个维度即x^i中图像信息所在的维度"""
    b = 0
    """为什么b是一维标量？"""
    """1.不需要修改，因为他只作用于边界，每次都不一样会增加学习难度"""
    """2.在计算y=wx+b的时候因为 numpy 广播机制（boardcast）可以自动扩展为 wx维度一样的矩阵"""

    # 使用断言来确保我要的数据是正确的
    # w 的维度是 (dim,1)
    assert (w.shape == (dim, 1)),"w 的维度不对"
    # b 的类型是 float 或者是 int
    assert (isinstance(b, float) or isinstance(b, int)),"b 不是标量"
    """断言（assert）不符合断言条件，会抛出断言错误"""

    return w, b

#w1,b1 = initialize_with_zeros(12288)
#print(w1.T,b1)
"""根据矩阵乘法，wx  先要将w转置，这样变成一个一列 dim个数的矩阵，其目的是和x^i矩阵维度对应"""
"""x^[i]为12280行，209列， （w dot x^[i]） =>> x第一行12280个分别与x ^[i]第一列 12280个乘，"""

#3.propage函数(forward + backword)
def propagate(w,b,x,Y):
    """
       实现前向和后向传播的传播函数，计算成本函数及其梯度。
       参数：
           w  - 权重，大小不等的数组（num_px * num_px * 3，1）
           b  - 偏差，一个标量
           X  - 矩阵类型为（num_px * num_px * 3，训练数量）
           Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

       返回：
           cost- 逻辑回归的负对数似然成本
           dw  - 相对于w的损失梯度，因此与w相同的形状
           db  - 相对于b的损失梯度，因此与b的形状相同
       """

    m = x.shape[1]
    """m == 209"""
    """显然，一次计算的是数据集所有图片的cost"""

    # 前向传播-----------------------------------------------
    z =  np.dot(w.T,x) + b
    """自动广播了b"""
    A = sigmoid(z)

    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    #校对答案，这也是为什么这里唯一引入了“答案Y”
    """当然，这次0，1就可分类，但是如果用到字符识别，那y可用（hotspot）类似寄存器的存储方法，即list[0,1,0,0,0,0,0]类似的"""

    # 反向传播-----------------------------
    dw = 1 / m * np.dot(x, (A - Y).T)
    """dl/dw = dl/dA *  (da/dy) * (dy/dw)   ==>> 根据链式法则"""
    """ da/dw = a * (1-a) * x   忽略a*（1-a），等式右边可看作 x """
    """dl/dA = 即对loss方程求导"""


    db = 1 / m * np.sum(A - Y)
    """同理，根据链式法则：dl/db = dl/dA * dA / db (对b求导)"""
    """dA/db = dA/dy * dy/db, 即 a * (1-a) * 1，常数，同样忽略"""
    """因此就是dA/da 前面已经导过，就是sum(A-Y)"""

    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个字典，把 dw 和 db 保存起来。
    grads = {"dw": dw,
             "db": db}
    """dw就是k（w）也就是下降的斜率"""
    """db也是同理"""

    return grads, cost



# 测试一下 propagate 函数
"""
print("====================测试propagate====================")
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
""""""y为一维，包含209个’答案‘  """"""
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
"""""""输出二维，即对于两个类型的判断""""""
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
"""
#4.对于参数进行修改

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
       此函数通过运行梯度下降算法来优化w和b

       参数：
           w  - 权重，大小不等的数组（num_px * num_px * 3，1）
           b  - 偏差，一个标量
           X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
           Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
           num_iterations  - 优化循环的迭代次数
           learning_rate  - 梯度下降更新规则的学习率
           print_cost  - 每100步打印一次损失值

       返回：
           params  - 包含权重w和偏差b的字典
           grads  - 包含权重和偏差相对于成本函数的梯度的字典
           成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

       提示：
       我们需要写下两个步骤并遍历它们：
           1）计算当前参数的成本和梯度，使用propagate（）。
           2）使用w和b的梯度下降法则更新参数。
       """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate* db
        """根据学习率进行梯度下降"""

        if i % 100 == 0:
            costs.append(cost)
            if(print_cost):
                print(f"Cost after iteration {i}: {abs(cost)}")

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
              "db": db}

    return params, grads, costs

#测试一下 optimize 函数

#print("====================测试optimize====================")
#params, grads, costs = optimize(w1, b1, train_set_x, train_set_y, num_iterations= 2000, learning_rate = 0.009, print_cost = False)
#print ("w shape= " + str(params["w"].shape))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))
#print(costs)


#5. 运用已经训练好的网络进行预测---------------------------------------------------------------------------------------------

def predict(w, b, X):
    """
        使用学习逻辑回归参数 logistic(w，b) 预测标签是0还是1，

        参数：
            w  - 权重，大小不等的数组（num_px * num_px * 3，1）
            b  - 偏差，一个标量
            X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据

        返回：
            Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

        """
    #制作一个向量表，来存储和判断网络对于输出的结果
    Y_prediction = np.zeros((1, X.shape[1]),dtype=np.uint8)

    w = w.reshape(X.shape[0], 1)
    """防止接下来计算点乘报错"""

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # 将概率 a[0，i] 转换为实际预测 p[0，i]
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction

"""
#测试小程序
res = predict(params["w"],grads["db"],train_set_x)
print(res)
print(train_set_y)
sum_res = 0
for i in range(len(res[0])):
    if (res[0,i]) == (train_set_y[0,i]):
        sum_res += 1
print(f"acu = {sum_res/len(res[0])*100}%")
"""

#6.封装到 model()--------------------------------------------
#配置超参数
def model(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate = 0.09, num_eporch = 1000, print_cost = True):

    #初始化
    print(train_set_x.shape)
    w1, b1 = initialize_with_zeros(train_set_x.shape[0])

    #梯度下降
    parameters, grads, costs = optimize(w1, b1, train_set_x, train_set_y, num_eporch, learning_rate, print_cost)

    # 从“parameters”字典中检索参数w和b
    w = parameters["w"]
    b = parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, test_set_x)
    Y_prediction_train = predict(w, b, train_set_x)

    # 打印训练后的准确性
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_eporch}

    return d

#model(train_set_x, train_set_y, test_set_x, test_set_y,0.0006,5700,True)



