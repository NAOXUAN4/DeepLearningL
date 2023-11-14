import mlp
import numpy as np
import cv2
from lr_utils import load_dataset

#-----------------------------------------预处理--------------------------------------------------------

#1.找出数据的尺寸和维度（m_train，m_test，num_px等）
#2.重塑数据集，以使每个示例都是大小为（num_px * num_px * 3，1）的向量
#3.“标准化”数据

"""1.读取数据集"""
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
test_real = cv2.imread("E:\py object\DeepLearningL\mlp\datasets\\real_datas\\not_cat2.jpg")
test_real = cv2.resize(test_real,(64,64))

test_real = np.dstack([test_real] * 1)

test_real = test_real.reshape(1, -1).T
test_real = test_real.astype(np.float32)
test_real/=255

print(test_real)


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

mlp.model(train_set_x, train_set_y, test_real,[[0]],0.0006,5700,True)