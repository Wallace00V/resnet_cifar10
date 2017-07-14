# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:09:06 2017

@author: zwcong2
"""
import input_data
import models
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

    
batch_size = 128


X_train, Y_train, X_test, Y_test = input_data.load_data()
Xtr=input_data.Dataset(X_train).next_batch(batch_size)

'''
X = tf.placeholder("float", [batch_size, 32, 32, 3])
Y = tf.placeholder("float", [batch_size, 10])
learning_rate = tf.placeholder("float", [])

# ResNet Models
net = models.resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cross_entropy = -tf.reduce_sum(Y*tf.log(net))
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_op = opt.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#saver = tf.train.Saver()
#checkpoint_dir=''
checkpoint = tf.train.latest_checkpoint(".")
if checkpoint:
    print ("Restoring from checkpoint", checkpoint)
    saver.restore(sess, checkpoint)
else:
    print ("Couldn't find checkpoint to restore from. Starting over.")


for j in range (10):
    for i in range (0, 500, batch_size):
        feed_dict={
            X: Xtr.next_batch(batch_size), 
            Y: Y_train[i:i + batch_size],
            learning_rate: 0.001}
        _,loss=sess.run([train_op,cross_entropy], feed_dict=feed_dict)
        if i % 512 == 0:
            print ("training on image #%d" % i, 'lost :',loss)

for i in range (0, 1000, batch_size):
    if i + batch_size < 1000:
        acc = sess.run(accuracy,feed_dict={
            X: X_test[i:i+batch_size],
            Y: Y_test[i:i+batch_size]
        })
        #accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        print (acc)

sess.close()'''