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
    w = tf.Variable(tf.constant(5, dtype=tf.float32))
    loss = tf.square(w+1)
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(40):
            sess.run(train_step)
            w_val = sess.run(w)
            loss_val = sess.run(loss)
            print("After %s steps: w is %f, loss is %f." % (i, w_val,loss_val))