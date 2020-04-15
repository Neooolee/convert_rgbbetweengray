# -*- coding: utf-8 -*-
"""
Created on Mon May  13 16:57:36 2020

@author: Neoooli
"""
from __future__ import print_function

from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import numpy as np
import glob
from collections import OrderedDict

rgb_colors=OrderedDict([
    ("cloud",np.array([255,255,255],dtype=np.uint8)),
    ("cloud-free",np.array([0,0,0],dtype=np.uint8))]) 
 #输入shape=(w,h,c),将rgb的mask转换为(w,h)的灰度图，每个像素点的值就是类别：0，1，2，..再经label=tf.one_hot(graymap,classes),转换即可进行交叉熵计算。
def convert_rgb_to_label(rgb_mask,rgb_colors):
    label = (np.zeros(batch_rgb_mask.shape[:2])).astype(np.uint32)
    if len(batch_rgb_mask.shape)==4:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((batch_rgb_mask == np.asarray(rgb_values)).astype(int).sum(-1) == 3)
            label[match_pixs] = gray        
    else:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((batch_rgb_mask == np.asarray(rgb_values)).astype(int) == 1)
            label[match_pixs] = gray
    return label.astype(np.uint32)
#输入shape=(batch_size,w,h,c)
def convert_rgbs_to_label(batch_rgb_mask,rgb_colors):
    label = (np.zeros(batch_rgb_mask.shape[:3])).astype(np.uint32)
    if len(batch_rgb_mask.shape)==4:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((batch_rgb_mask == np.asarray(rgb_values)).astype(int).sum(-1) == 3)
            label[match_pixs] = gray        
    else:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((batch_rgb_mask == np.asarray(rgb_values)).astype(int) == 1)
            label[match_pixs] = gray
    return label.astype(np.uint32)
 #输入shape=(w,h,c)
def convet_label_to_rgb(label,rgb_colors):
    max_index=np.argmax(label,axis=2)#第三维上最大值的索引，返回其他维度，并在并对位置填上最大值之索引
    if label.shape[2]>=3:
        rgb = (np.zeros(label.shape[:2]+tuple([3]))).astype(np.uint8)
    else:
        rgb = (np.zeros(label.shape[:2])).astype(np.uint8)
    for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
        match_pixs = np.where(max_index == gray)  
        rgb[match_pixs] = rgb_values
    return rgb.astype(np.uint8)
#输入shape=(batch_size,w,h,c)
def convet_labels_to_rgb(batch_labels,rgb_colors):
    max_index=np.argmax(batch_labels,axis=3)#第三维上最大值的索引，返回其他维度，并在并对位置填上最大值之索引
    rgb = (np.zeros(batch_labels.shape[:3]+tuple([3]))).astype(np.uint8)
    for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
        match_pixs = np.where(max_index == gray)    
        rgb[match_pixs] = rgb_values
    return rgb.astype(np.uint8)
 
