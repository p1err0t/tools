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
class FunChap4:
    @staticmethod
    def func4_1():
        w = tf.Variable(tf.constant(5, dtype=tf.float32))
        loss = tf.square(w+1)
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(10000):
                sess.run(train_step)
                w_val = sess.run(w)
                loss_val = sess.run(loss)
                print("After %s steps: w is %f, loss is %f." % (i, w_val,loss_val))
    @staticmethod
    def func4_2():
        LEARNING_RATE_BASE = 0.1 #初始学习率
        LEARNING_RATE_DECAY = 0.99 #学习率衰减率
        LEARNING_RETE_STEP = 1 #喂入多少轮BATCH_SIZE后，更新一次学习率，一般为：总样本数/BATCH_SIZE

        #运行了几轮BATCH_SIZE的计算器，初值为0，设为不被训练
        global_step = tf.Variable(0, trainable=False)
        # 定义指数下降学习率
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                                   LEARNING_RETE_STEP,LEARNING_RATE_DECAY,staircase=True)
        w = tf.Variable(tf.constant(5, dtype=tf.float32))
        loss = tf.square(w+1)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(40):
                sess.run(train_step)
                learning_rate_val = sess.run(learning_rate)
                global_step_val = sess.run(global_step)
                w_val = sess.run(w)
                loss_val = sess.run(loss)
                print("After %s steps: global_step is %f, w is %f,learning rate is %f,"
                      " loss is %f." % (i,global_step_val, w_val,learning_rate_val,loss_val))


FunChap4.func4_2()