import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
这个命令可以控制python 程序显示提示信息的等级，在Tensorflow 里面一般设
置成是"0"（显示所有信息）或者"1"（不显示info），"2"代表不显示warning，
"3"代表不显示error。一般不建议设置成3。
'''
import tensorflow as tf
import numpy as np

#开启动态图
#tf.enable_eager_execution()

class learnTf:
    @staticmethod
    def tf1():
        A = tf.constant([[1, 2], [3, 4]])
        B = tf.constant([[1, 2], [7, 8]])
        C = tf.matmul(A, B)
        print(C)

    @staticmethod
    def tf2():
        x = tf.constant([[1, 2], [3, 4]])
        w = tf.constant([[1, 2], [7, 8]])
        y = tf.matmul(x, w)
        print(y)
        with tf.Session() as sess:
            print(sess.run(y))
    '''
    神经网络Neural Network,NN
    '''
    # 两层简单神经网络（全连接）
    @staticmethod
    def tf3():
        # 定义输入和参数
        x = tf.constant([[0.7, 0.5]])
        w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

        # 定义前向传播过程
        a = tf.matmul(x, w1)
        y = tf.matmul(a, w2)

        # 用会话计算结果
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("y in fun_tf3 is :\n", sess.run(y))

    @staticmethod
    def tf4():
        # 定义输入和参数
        # 用placehold实现输入定义（sess.run中喂一组数据）
        x = tf.placeholder(tf.float32, shape=(1, 2))
        w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

        # 定义前向传播过程
        a = tf.matmul(x, w1)
        y = tf.matmul(a, w2)

        # 用会话计算结果
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("y in fun_tf4 is :\n", sess.run(y, feed_dict={x: [[0.7, 0.5]]}))


    def tf5(self):
        # 定义输入和参数
        # 用placehold实现输入定义（sess.run中喂多组数据）
        x = tf.placeholder(tf.float32, shape=(None, 2))
        w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

        # 定义前向传播过程
        a = tf.matmul(x, w1)
        y = tf.matmul(a, w2)

        # 用会话计算结果
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("y in fun_tf4 is :\n",
                  sess.run(y, feed_dict={x: [[0.7, 0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
            print("w1:\n", sess.run(w1))
            print("w2:\n", sess.run(w2))

    def tf6(self):
        BATCH_SIZE = 8
        seed = 23455
        # 基于seed产生随机数
        rng = np.random.RandomState(seed)
        # 随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
        X = rng.rand(32, 2)
        # 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1，否则为0
        # 作为输入数据集的标签
        Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
        print("X:\n", X)
        print("Y:\n", Y)

        # 1定义神经网络的输入、参数和输出，以及前向传播过程
        x = tf.placeholder(tf.float32, shape=(None, 2))
        y_ = tf.placeholder(tf.float32, shape=(None, 1))
        w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
        a = tf.matmul(x, w1)
        y = tf.matmul(a, w2)

        # 2定义损失函数及反向传播方法
        loss = tf.reduce_mean(tf.square(y-y_))
        # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        # train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        # 3生成会话，训练STEPS轮
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("w1:\n", sess.run(w1))
            print("w2:\n", sess.run(w2))
            print("\n")

            #训练模型
            STEPS = 5000
            for i in range(STEPS):
                start = (i*BATCH_SIZE) % 32
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
                if i % 500 == 0:
                    total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
                    print("After %d training step(s), loss on all data is %g" %(i, total_loss))
            print("\n")
            print("w1:\n", sess.run(w1))
            print("w2:\n", sess.run(w2))







runit = learnTf()
runit.tf6()
