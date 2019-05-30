import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
这个命令可以控制python 程序显示提示信息的等级，在Tensorflow 里面一般设
置成是"0"（显示所有信息）或者"1"（不显示info），"2"代表不显示warning，
"3"代表不显示error。一般不建议设置成3。
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

    @staticmethod
    def func4_3():
        w1 = tf.Variable(0, dtype=tf.float32)
        # 定义num_updates(NN的迭代轮数)，初始值为0，不可被优化
        global_step = tf.Variable(0, trainable=False)
        # 实例化滑动平均类，滑动平均衰减率为0.99
        MOVING_AVERAGE_DECAY = 0.99
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        # 更新列表，求滑动平均值
        ema_op = ema.apply(tf.trainable_variables())
        # 查看不同迭代中变量取值的变化
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print(sess.run([w1, ema.average(w1)]))

            # 参数w1的值赋为1
            sess.run(tf.assign(w1, 1))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))

            # 更新step和w1的值，模拟出100轮迭代后，参数w1变为10
            sess.run(tf.assign(global_step, 100))
            sess.run(tf.assign(w1, 10))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))

            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))
            sess.run(ema_op)
            print(sess.run([w1, ema.average(w1)]))

    @staticmethod
    def get_weight(shape, regularizer):
        w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w
    @staticmethod
    def get_bias(shape):
        b = tf.Variable(tf.constant(0.01, shape = shape))
        return b


    @classmethod
    def func4_4(self):
        BATCH_SIZE = 30
        seed = 2
        rdm = np.random.RandomState(seed)
        #randn标准正态分布
        X = rdm.randn(300, 2)
        Y_ = [int(x0*x0 + x1*x1 < 2) for(x0, x1) in X]
        Y_c = [['red' if y else 'blue'] for y in Y_]
        #Y_c = ['red' if y else 'blue' for y in Y_]
        # 把X整理为n行2列，Y_整理为n行1列
        X = np.vstack(X).reshape(-1, 2)
        Y_ = np.vstack(Y_).reshape(-1, 1)
        print(X)
        print(Y_)
        print(Y_c)
        plt.scatter(X[:,0],X[:,1], c=np.squeeze(Y_c))
        plt.show()

        x = tf.placeholder(tf.float32, shape=[None,2])
        y_ = tf.placeholder(tf.float32, shape=[None,1])

        w1 = self.get_weight([2,11], 0.01)
        b1 = self.get_bias([11])
        y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

        w2 = self.get_weight([11,1], 0.01)
        b2 = self.get_bias([1])
        y = tf.matmul(y1, w2) + b2

        loss_mse = tf.reduce_mean(tf.square(y-y_))
        loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            STEPS = 40000
            for i in range(STEPS):
                start = (i*BATCH_SIZE) % 300
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={x:X[start:end],y_:Y_[start:end]})
                if i % 2000 == 0:
                    loss_mse_v = sess.run(loss_mse, feed_dict={x:X,y_:Y_})
                    print('After %d steps,loss is %f.'%(i,loss_mse_v))
            xx,yy = np.mgrid[-3:3:.01, -3:3:.01]
            grid = np.c_[xx.ravel(),yy.ravel()]
            probs= sess.run(y, feed_dict={x:grid})
            probs= probs.reshape(xx.shape)
            print('w1:\n',sess.run(w1))
            print('b1:\n',sess.run(b1))
            print('w2:\n',sess.run(w2))
            print('b2:\n',sess.run(b2))

        plt.scatter(X[:,0],X[:,1], c=np.squeeze(Y_c))
        plt.contour(xx,yy,probs,levels=[.5])
        plt.show()

        #包含正则化
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            STEPS = 40000
            for i in range(STEPS):
                start = (i*BATCH_SIZE)%300
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={x:X[start:end],y_:Y_[start:end]})
                if i % 2000 == 0:
                    loss_mse_v = sess.run(loss_mse, feed_dict={x:X,y_:Y_})
                    print('After %d steps,loss is %f.'%(i,loss_mse_v))
            xx,yy = np.mgrid[-3:3:.01, -3:3:.01]
            grid = np.c_[xx.ravel(),yy.ravel()]
            probs= sess.run(y, feed_dict={x:grid})
            probs= probs.reshape(xx.shape)
            print('w1:\n',sess.run(w1))
            print('b1:\n',sess.run(b1))
            print('w2:\n',sess.run(w2))
            print('b2:\n',sess.run(b2))

        plt.scatter(X[:,0],X[:,1], c=np.squeeze(Y_c))
        plt.contour(xx,yy,probs,levels=[.5])
        plt.show()






FunChap4.func4_4()