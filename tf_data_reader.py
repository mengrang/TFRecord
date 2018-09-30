# -*- coding:utf-8 -*-
import json
import os
import tensorflow as tf
import random
import cv2
import numpy as np
import math

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

# 根据json读取图片，pading成448**2写入tfrecord
def pading_write_tfrecord(data_dir, split_dir, json_file):
    # Args
    split_path = os.path.join(data_dir, split_dir)
    image_path = os.path.join(split_path, 'images')
    # jsonname = 'AgriculturalDisease_train_annotations.json'
    # load json
    # print(os.path.join(split_path, json_file))
    # exit()
    # path_to_json = os.path.join(split_path, json_file)
    # path_to_json = path_to_json.replace('\\','\')
    with open(os.path.join(split_path, json_file), 'r', encoding='utf-8') as f:
        py_data = json.load(f)
    tfrecord_name = os.path.splitext(json_file)[0]
    tfrecord_file = tfrecord_name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    cout = 0
    for py_dict in py_data:
        disease_class = py_dict['disease_class']
        image_id = py_dict['image_id']
        img = cv_imread(os.path.join(image_path, image_id))
        
        # pading
        # img_shape = img.shape
        img_h = img.shape[0]
        img_w = img.shape[1]
        if max(img_h, img_w) >= 448:
            ratio = max(img_h, img_w) / 447
            img = cv2.resize(img, (math.ceil(img_h//ratio), math.ceil(img_w//ratio)))
            img_h = img.shape[0]
            img_w = img.shape[1]
        # print(img.shape)
        # exit() 
        img_pad = np.zeros((448, 448, 3), dtype=np.float32)
        h_pad_beg = (448 - img_h) // 2
        w_pad_beg = (448 - img_w) // 2
        h_pad_end = h_pad_beg + img_h
        w_pad_end = w_pad_beg + img_w  
        img_pad[h_pad_beg:h_pad_end, w_pad_beg:w_pad_end, :] = img[:, :, :]
        # img_pad = cv2.copyMakeBorder(img, h_pad_beg, h_pad_end, w_pad_beg, w_pad_end, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img_raw = img_pad.tobytes()                                  #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[disease_class])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())                   #序列化为字符串
        cout = cout + 1
        print(cout)
    writer.close()
    print('The tfrecord writing end...')
    return py_data, tfrecord_file

def read_and_decode(tfrecord_file):
    #根据tfrecord文件名生成一个队列
    # tfrecord_name = os.path.splitext(json_file)[0]
    # tfrecord_file_name = tfrecord_name + '.tfrecords'
    # if os.path.exists(os.path.join(data_dir, split_dir, tfrecord_file):
    #     tfrecord_file = 
    # _, tfrecord_file = pading_write_tfrecord(data_dir, split_dir, json_file)
    tfrecord_queue = tf.train.string_input_producer([tfrecord_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                    features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw' : tf.FixedLenFeature([], tf.string),
                                    })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [448, 448, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.float32)

    return img, label



# img, label = read_and_decode("AgriculturalDisease_trainingset.tfrecords")
# #使用shuffle_batch可以随机打乱输入, 正常
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=30, capacity=2000,
#                                                 min_after_dequeue=1000)
if __name__=='__main__':
    pading_write_tfrecord('E:\plant', 'AgriculturalDisease_trainingset', 'AgriculturalDisease_train_annotations.json')
