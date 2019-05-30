import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
这个命令可以控制python 程序显示提示信息的等级，在Tensorflow 里面一般设
置成是"0"（显示所有信息）或者"1"（不显示info），"2"代表不显示warning，
"3"代表不显示error。一般不建议设置成3。
'''
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
#开启动态图
#tf.enable_eager_execution()

class Tensor_f4:
    @staticmethod
    def func4_1():
        seed = 23455
        # 基于seed产生随机数
        rdm = np.random.RandomState(seed)
        # 随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
        X = rdm.rand(32, 2)
        # 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1，否则为0
        # 作为输入数据集的标签
        Y_ = [[x1+x2+(rdm.rand()/10-0.05)] for (x1, x2) in X]
        # print("X:\n", X)
        # print("Y:\n", Y_)

        # 1定义神经网络的输入、参数和输出，以及前向传播过程
        x = tf.placeholder(tf.float32, shape=(None, 2))
        y_ = tf.placeholder(tf.float32, shape=(None, 1))
        w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
        # w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
        # a = tf.matmul(x, w1)
        y = tf.matmul(x, w1)

        # 2定义损失函数及反向传播方法
        loss = tf.reduce_mean(tf.square(y-y_))
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        # train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
        # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        # 3生成会话，训练STEPS轮
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # print("w1:\n", sess.run(w1))
            # print("w2:\n", sess.run(w2))
            # print("\n")

            #训练模型
            STEPS = 20000
            for i in range(STEPS):
                start = (i*BATCH_SIZE) % 32
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
                if i % 1000 == 0:
                    total_loss = sess.run(loss, feed_dict={x: X, y_: Y_})
                    print("After %d training step(s), loss on all data is %g" % (i, total_loss))
                    print('w1 is:\n',sess.run(w1))
            print("\n")
            print("Final w1 is:\n", sess.run(w1))
            # print("w2:\n", sess.run(w2))

    @staticmethod
    def func4_2():
        seed = 23455
        COST = 9
        PROFIT = 1
        # 基于seed产生随机数
        rdm = np.random.RandomState(seed)
        # 随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
        X = rdm.rand(32, 2)
        # 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1，否则为0
        # 作为输入数据集的标签
        Y_ = [[x1+x2+(rdm.rand()/10-0.05)] for (x1, x2) in X]
        # print("X:\n", X)
        # print("Y:\n", Y_)

        # 1定义神经网络的输入、参数和输出，以及前向传播过程
        x = tf.placeholder(tf.float32, shape=(None, 2))
        y_ = tf.placeholder(tf.float32, shape=(None, 1))
        w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
        # w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
        # a = tf.matmul(x, w1)
        y = tf.matmul(x, w1)

        # 2定义损失函数及反向传播方法
        # loss = tf.reduce_mean(tf.square(y-y_))
        loss = tf.reduce_sum(tf.where(tf.greater(y,y_),COST*(y-y_),PROFIT*(y_-y)))
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        # train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
        # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        # 3生成会话，训练STEPS轮
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # print("w1:\n", sess.run(w1))
            # print("w2:\n", sess.run(w2))
            # print("\n")

            #训练模型
            STEPS = 20000
            for i in range(STEPS):
                start = (i*BATCH_SIZE) % 32
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
                if i % 1000 == 0:
                    total_loss = sess.run(loss, feed_dict={x: X, y_: Y_})
                    print("After %d training step(s), loss on all data is %g" % (i, total_loss))
                    print('w1 is:\n',sess.run(w1))
            print("\n")
            print("Final w1 is:\n", sess.run(w1))
            # print("w2:\n", sess.run(w2))
    @staticmethod
    def func4_4():
        w = tf.Variable(tf.constant(5,dtype=tf.float32))
        loss = tf.square(w+6)
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(40):
                sess.run(train_step)
                w_val = sess.run(w)
                loss_val = sess.run(loss)
                print("After %s steps: w is %f, loss is %f." %(i,w_val,loss_val))


Tensor_f4.func4_4()


