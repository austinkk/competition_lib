#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import io
import os
import sys

class ImageTool(object):
    def __init__(self):
        pass

    def open_image(self, filename, mode = 'r'):
        # mode  '1': 1位像素，黑白图像，存成8位像素。 'L': 8位像素，黑白。 'P': 9位像素，使用调色板映射到任何其他模式
        # 'RGB' : 3 * 8位像素，真彩    'RGBA': 4 * 8位像素，真彩+透明通道  'CMYK': 4*8位像素，印刷四色模式或彩色印刷模式
        # 'YCbCr' : 3*8位像素，色彩视频格式    'I': 32位整型像素    'F':32位浮点型像素
        im = Image.open(filename, mode)
        print ('image format: %s, size: %s, mode ' % (im.format, str(im.size), im.mode))
        print ('image bands: %s' % str(im.getbands()))
        print ('image box: %s' % str(im.getbbox()))

    def save_image(self, im, filename):
        im.save(filename)

    def convert(self, im, mode):
        return im.convert(mode)
    
    def copy(self, im):
        return im.copy()
    
    # 裁剪图片
    def crop(self, im, box):
        # box 为 (x1, y1, x2, y2) 四元组，左上右下坐标
        return im.crop(box)
    
    # 将一张图粘贴到另一张图上
    def paste(self, im1, im2, box):
        # im2粘贴到im1上，box大小要和im2大小一样
        im1.paste(im2, box)

    # 使用指定滤波器处理图像
    def filter_image(self, im, imagefilter):
        # ImageFilter.BLUR 均值滤波
        # ImageFilter.CONTOUR 找轮廓
        # ImageFilter.FIND_EDGES 边缘检测
        # BLUR、CONTOUR、DETAIL、EDGE_ENHANCE、EDGE_ENHANCE_MORE、EMBOSS、FIND_EDGES、SMOOTH、SMOOTH_MORE、SHARPEN
        return im.filter(imagefilter)

    # 使用两张图像及透明度变量alpha，插出一张新的图像
    def blend(self, im1, im2, alpha = 0.4):
        return Image.blend(im1, im2, alpha)
    
    # 返回当前图像各个通道组成的一个元组。例如，分离一个“RGB”图像将产生三个新的图像，分别对应原始图像的每个通道（红，绿，蓝）
    def split(self, im1):
        r, g, b = im.split()
        return r, g, b
    
    # 复合类使用给定的两张图像及mask图像作为透明度，插值出一张新的图像。
    # 变量mask图像的模式可以为“1”，“L”或者“RGBA”。所有图像必须有相同的尺寸
    def composite(self, im1, im2, mask):
        return Image.composite(im1, im2, mask)
    
    # 使用变量function对应的函数（该函数应该有一个参数）处理变量image所代表图像中的每一个像素点。
    # 如果变量image所代表图像有多个通道，那变量function对应的函数作用于每一个通道。
    def eval_func(self, im, func):
        return Image.eval(im,func)
    
    # 合并类使用一些单通道图像，创建一个新的图像。
    # 变量bands为一个图像的元组或者列表，每个通道的模式由变量mode描述。所有通道必须有相同的尺寸。
    def merge(self, mode = "RGB", new_im):
        # new_im = [r, g, b]
        return  Image.merge(mode, new_im)
    
    # 关键信息显示
    def draft(self, im, mode, box):
        return im.draft(mode, box)

    # 以包含像素值的sequence对象形式返回图像的内容。
    def getdata(self, im):
        return list(im.getdata())
    
    # 返回了R/G/B三个通道的最小和最大值的2元组。
    def getextrema(self, im):
        return im.getextrema()
    
    # 返回给定位置的像素值
    def getpixel(self, im, x, y):
        return im.getpixel((x, y))

    # 返回一个图像的直方图。这个直方图是关于像素数量的list，图像中的每个象素值对应一个成员。
    # 如果图像有多个通道，所有通道的直方图会连接起来。
    def gethistogram(self, im):
        return im.histogram()
    
    # size 是个二元组 fil 为 NEAREST、BILINEAR、BICUBIC或者ANTIALIAS之一
    # PIL.Image.NEAREST (use nearest neighbour), PIL.Image.BILINEAR (linear interpolation)
    # PIL.Image.BICUBIC (cubic spline interpolation), or PIL.Image.LANCZOS (a high-quality downsampling filter)
    def resize(self, im, size,  fil = Image.NEAREST):
        return im.resize(size, fil)

    # PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_90
    # PIL.Image.ROTATE_180, PIL.Image.ROTATE_270 or PIL.Image.TRANSPOSE
    def transpose(self, im, method):
        return im.transpose()

    #  PIL.Image.NEAREST (use nearest neighbour), PIL.Image.BILINEAR (linear interpolation in a 2x2 environment)
    #  PIL.Image.BICUBIC (cubic spline interpolation in a 4x4 environment). 
    # if omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
    # expand 0: same size to original pic 1: expand the picture
    def rotate(self, im, angle,  resample=0, expand=0):
        return im.rotate(angle, resample, expand)
    
    # TODO
    #def transform

class ImageTFRecorder(object):
    def __init__(self):
        self.images_dir = './data/..'
        self.annotation_path = './data/..'
        self.record_path = './data/..'
    
    def int64_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    
    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))
    
    def process_image_channels(self, image):
        process_flag = False
        # process the 4 channels .png
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
            process_flag = True
        # process the channel image
        elif image.mode != 'RGB':
            image = image.convert("RGB")
            process_flag = True
        return image, process_flag

    def process_image_reshape(self, image, resize):
        width, height = image.size
        if resize is not None:
            if width > height:
                width = int(width * resize / height)
                height = resize
            else:
                width = resize
                height = int(height * resize / width)
        return image
    
    def create_tf_example(self, image_path, label, resize = None):
        with tf.gfile.GFile(image_path, 'rb') as fid:
            encode_jpg = fid.read()
        encode_jpg_io = io.BytesIO(encode_jpg)
        image = Image.open(encode_jpg_io)
        # process png pic with four channels
        image, process_flag = self.process_image_channels(image)
        # reshape image
        image = self.process_image_reshape(image, resize)
        if process_flag == True or resize is not None:
            bytes_io = io.BytesIO()
            image.save(bytes_io, format = 'JPEG')
            encode_jpg = bytes_io.getvalue()
        width, height = image.size
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': self.bytes_feature(encode_jpg),
                    'image/format': self.bytes_feature(b'jpg'),
                    'image/class/label': self.int64_feature(label),
                    'image/height': self.int64_feature(height),
                    'image/width': self.int64_feature(width)
                }
            )
        )
        return tf_example

    def generate_tfrecord(self, annotation_list, record_path, resize=None):
        num_tf_example = 0
        writer = tf.python_io.TFRecordWriter(record_path)
        for image_path, label in annotation_list:
            if not tf.gfile.GFile(image_path):
                print("{} does not exist".format(image_path))
            tf_example = self.create_tf_example(image_path, label, resize)
            writer.write(tf_example.SerializeToString())
            num_tf_example += 1
            if num_tf_example % 100 == 0:
                print("Create %d TF_Example" % num_tf_example)
        writer.close()
        print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

