import numpy as np
import matplotlib.pyplot as plt
from reg_utils import *
import sklearn
import sklearn.datasets
#from testCase import *



# 设置 plot 的默认大小
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.show()

train_X, train_Y, test_X, test_Y = load_2D_dataset(is_plot=True)

