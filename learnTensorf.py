import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.enable_eager_execution()

class learnTf:
    @staticmethod
    def tf1():
        A = tf.constant([[1, 2], [3, 4]])
        B = tf.constant([[1, 2], [7, 8]])
        C = tf.matmul(A, B)
        print(C)


learnTf.tf1()
