
# coding: utf-8

# # 变量管理

import tensorflow as tf

# # tf.get_variable(name, shape, initializer) 变量名必填

# tf.constant_initializer
# 变量初始化成常数    args: 常量取值
tf.constant_initializer(1.0)

# tf.random_normal_initializer
# 将变量初始化为满足正态分布的随机值    args: 正太分布均值和标准差

# tf.truncated_normal_initializer
# 将变量初始化为满足正态分布的随机值, 但如果随机出来的值偏离平均值超过2个标准差,那么这个数将会被重新随机    args: 正太分布均值和标准差

# tf.random_uniform_initializer
# 将变量初始化为满足平均分布的随机值    args: 最大最小值

# tf.uniform_unit_scaling_initializer
# 将变量初始化为满足平均分布但不影响输出量级的随机值    arg: factor(产生随机值时乘以系数`)

# tf.zeros_initializer
# 将变量全部设置为0    arg 变量维度

# tf.ones_initializer
# 将变量全部设置为1    arg 变量维度


# # tf.Variable(initializer, name) 变量名选填


# tf.random_normal
# 正太分布    args: 平均值,标准差,取值类型

# tf.truncated_normal
# 将变量初始化为满足正态分布的随机值, 但如果随机出来的值偏离平均值超过2个标准差,那么这个数将会被重新随机    args: 平均值,标准差,取值类型
tf.Variable(tf.truncated_normal([11, 11, 3, 64], tf.float32, stddev=1e-1), name = 'weights') # AlexNet

# tf.random_uniform
# 平均分布

# tf.random_gamma
# Gamma 分布    args: 形状参数 alpha,尺度参数 beta, 取值类型

# tf.zeros
# tf.zeros([2, 3], int32) -> [[0, 0, 0], [0, 0, 0]]

# tf.ones
# tf.ones([2, 3], int32) -> [[1, .1, 1], [1, 1, 1]]

# tf.fill
# tf.fill([2, 3], 9) -> [[9, 9, 9], [9, 9, 9]]

# tf.constant
# tf.constant([1, 2, 3]) -> [1, 2, 3]


# # tf.variable_scope (要共享变量必须用这个)  配合 tf.get_variable用
# # tf.name_scope 配合 tf.Variable用,可以自动判断变量名是否重复
# # tf.variable_scope可以给tf.Variable用,但是tf.name_scope对tf.get_variable无效


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))


print (v) # ':0'是节点输出的第一个结果



with tf.variable_scope("foo", reuse= True):
     v = tf.get_variable("v", [1], initializer=tf.constant_initializer(2.0))


# variable_scope 嵌套
with tf.variable_scope("root"):
    print (tf.get_variable_scope().reuse)
    with tf.variable_scope("foo", reuse = True):
        print (tf.get_variable_scope().reuse)
        with tf.variable_scope("bar"):
            print (tf.get_variable_scope().reuse)


v1 = tf.get_variable("v", [1])
print (v1)
with tf.variable_scope("a"):
    v2 = tf.get_variable("v", [1])
    print (v2)
    with tf.variable_scope("b"):
        v3 = tf.get_variable("v", [1])
        print (v3)


# get variable scope
with tf.variable_scope("", reuse=True):
    print (tf.get_variable("v", [1]))
    print (tf.get_variable("a/v", [1]))
    print (tf.get_variable("a/b/v", [1]))

with tf.variable_scope("a/b", reuse=True):
    print (tf.get_variable("v", [1]))


# 无则创建,有则共享
with tf.variable_scope('scp', reuse=tf.AUTO_REUSE) as scp:    
    a = tf.get_variable('a') #无，创造
    a = tf.get_variable('a') #有，共享


# # TensorFlow模型持久化

# 声明网络结构后
# ...
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# 训练时
# model.ckpt.meta 神经网络结构
# model.ckpt 网络参数
# checkpoint 一个目录下所有模型文件列表
with tf.Session() as sess:
    sess.run(init_op)
    saver.saver(sess, "/path/to/model/model.ckpt") # 实际产生3个文件

# 声明网络结构后
# ...
#init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# 训练时
# model.ckpt.meta 神经网络结构
# model.ckpt 网络参数
# checkpoint 一个目录下所有模型文件列表
with tf.Session() as sess:
    #sess.run(init_op)
    saver.restore(sess, "/path/to/model/model.ckpt")
    saver.saver(sess, "/path/to/model/model.ckpt") # 实际产生3个文件

# 加载模型结构
saver = tf.train.import_meta_graph("/path/to/model/model.ckpt/model.ckpt.meta")
#init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# 加载部分变量 saver = tf.train.Saver({"v1": v1, "v2": v2})

# 加载滑动平均变量
# ema = tf.train.ExponentialMovingAverage(0.99)
# saver = tf.train.Saver(ema.variables_to_restore())

# 训练时
# model.ckpt.meta 神经网络结构
# model.ckpt 网络参数
# checkpoint 一个目录下所有模型文件列表
with tf.Session() as sess:
    #sess.run(init_op)
    saver.restore(sess, "/path/to/model/model.ckpt")
    print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))

