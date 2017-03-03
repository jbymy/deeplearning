#coding=utf-8 
'''
    常量赋值：hello = tf.constant('Hello,world!', dtype=tf.string)
    变量赋值：a = tf.Variable(10, dtype=tf.int32)
    数据类型：
        tf.int8：8位整数。
        tf.int16：16位整数。
        tf.int32：32位整数。
        tf.int64：64位整数。
        tf.uint8：8位无符号整数。
        tf.uint16：16位无符号整数。
        tf.float16：16位浮点数。
        tf.float32：32位浮点数。
        tf.float64：64位浮点数。
        tf.double：等同于tf.float64。
        tf.string：字符串。
        tf.bool：布尔型。
        tf.complex64：64位复数。
        tf.complex128：128位复数。
'''

 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import csv 
import numpy  
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
FLAGS = None
rng = numpy.random  

def readdense(feature_num):
    #读取数据
    rawData = []
    file = csv.reader(open("feature.txt"))
    linenum = 0;
    for line in file:
        rawData.append(line)
        linenum += 1
    data = numpy.array(rawData).astype(numpy.float32)    
    #创建tensor数据类型
    tensoedata = (rng.randn(linenum, feature_num), rng.randn(linenum, 2))
    #numpy 转tensor数据
    for i in range(linenum):
        tensoedata[0][i] = data[i][0:feature_num]
        tensoedata[1][i] = data[i][feature_num:feature_num+2]
    return (tensoedata,linenum)

def main(_):
    #Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # Define loss and optimizer
    feature_num = 100
    traindata,number = readdense(feature_num)
    x = tf.placeholder(tf.float32, [None, feature_num])
    W = tf.Variable(tf.zeros([feature_num, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])
  
    # The raw formulation of cross-entropy,
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    # can be numerically unstable.
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    # 计算平均交叉熵  
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #梯度下降算法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs = traindata[0]
        batch_ys = traindata[1]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        #import time
        #time.sleep(10)

    #结果评估 tf.argmax :给出某个tensor对象在某一维上的其数据最大值所在的索引值。
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: traindata[0],
                                      y_: traindata[1]}))
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
