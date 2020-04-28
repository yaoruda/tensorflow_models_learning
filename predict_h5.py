#coding=utf-8
from PIL import Image
# import Ipynb_importer 
import tensorflow as tf 
import numpy as np 
import pdb
import os
import glob
import slim.nets.mobilenet_v1 as mobilenet_v1

import tensorflow.contrib.slim as slim

# import matplotlib
# matplotlib.use('Agg')
from keras.models import load_model
# import cv2


def read_image_self(filename, resize_height, resize_width,normalization=False):

    
    image = Image.open(filename)
    image = image.resize([resize_height, resize_width])
    image_array = np.array(image)
    image_array = image_array.astype(float)
    rgb_image = np.reshape(image_array,[resize_height,resize_width, 3])
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)

    return rgb_image
 
def predict(models_path,image_dir,labels_filename,labels_nums, data_format):
    #加载模型h5文件
    model = load_model("models/net.h5")
    model.summary()


    [batch_size, resize_height, resize_width, depths] = data_format
 
    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
 
    #其他模型预测请修改这里
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        out, end_points = mobilenet_v1.mobilenet_v1(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)
 
    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)
 
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    #print(images_list)
    for image_path in images_list:
        #print(image_path)
        im=read_image_self(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score))
    sess.close()
 
 
if __name__ == '__main__':
    model = load_model("models/net.h5")
    model.summary()
    
#     class_nums=5
#     image_dir='test_image'
#     labels_filename='dataset/label.txt'
#     models_path='models/model.ckpt-60000'
 
#     batch_size = 1  #
#     resize_height = 244  # 指定存储图片高度
#     resize_width = 244  # 指定存储图片宽度
#     depths=3
#     data_format=[batch_size,resize_height,resize_width,depths]
#     predict(models_path,image_dir, labels_filename, class_nums, data_format)
