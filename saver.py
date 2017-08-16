import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
# remember to define the same dtype and shape when restore
#W = tf.Variable([[1,2,3],[3,4,5]],dtype = tf.float32,name='weights')
#b = tf.Variable([[1,2,3]],dtype=tf.float32,name='bias')

#init = tf.global_variables_initializer()

#saver = tf.train.Saver()

#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess,'my_net/save_net.ckpt')
#    print('save to path:',save_path)



W = tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32,name='bias')
# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'my_net/save_net.ckpt')
    print('weights:', sess.run(W))
    print('bias:', sess.run(b))