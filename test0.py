# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:09:06 2017

@author: zwcong2
"""
import input_data
import models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 128
learning_rate = 0.001
training_epochs =10
#读取图片和标签
X_train, Y_train, X_test, Y_test = input_data.load_data()

#train
X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

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


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys=input_data.DataSet(X_train,Y_train).next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op,cross_entropy], 
                            feed_dict={X: batch_xs,
                            Y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # Test model
    print("Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    # Save model
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)    
    
    
    
    
    