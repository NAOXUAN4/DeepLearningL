import lr_utils
import muiltnn

nn = muiltnn

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = nn.L_layer_model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True,isPlot=True,isReg=True)

pred_train = nn.predict(train_x, train_y, parameters)  # 训练集
pred_test = nn.predict(test_x, test_y, parameters)  # 测试集