#!/usr/bin/env python
# coding: utf-8

# Gfile API
import tensorflow as tf

def TFGfile(object):
    def __init__(self):
        pass
    
    # copy
    def copy_file(self, oldpath, newpath, overwrite = False):
        tf.gfile.Copy(oldpath, newpath, overwrite)
    
    # mkdir
    def mkdir(self, dirname):
        tf.gfile.MkDir(dirname)

    # rm
    def remove(self, filename):
        tf.gfile.Remove(filename)

    # rm -rf
    def delete_dir(self, dirname):
        tf.gfile.DeleteRecursively(dirname)
    
    # 判断目录或文件是否存在
    def file_exist(self, filename):
        return tf.gfile.Exists(filename)
    
    # 判断目录是否存在
    def is_directory(dirname):
        return tf.gfile.IsDirectory(dirname)

    # filename 可以是一个具体的文件名，也可以是包含通配符的正则表达式
    def search_file_list(self, filename):
        return tf.gfile.Glob(filename)

    # 罗列dirname目录下的所有文件并以列表形式返回，dirname必须是目录名
    def list_directory(self, dirname):
        return tf.gfile.ListDirectory(dirname)
    
    # 以递归方式建立父目录及其子目录，如果目录已存在且是可覆盖则会创建成功，否则报错，无返回
    def makedirs(self, dirname):
        return tf.gfile.MakeDirs(dirname)
    
    # mv a b
    def rename(self, oldname, newname, overwrite = False):
        tf.gfile.Rename(oldname, newname, overwrite)
    
    # 递归获取目录信息生成器，top是目录名，in_order默认为True指示顺序遍历目录，否则将无序遍历
    # 每次生成返回如下格式信息(dirname, [subdirname, subdirname, ...], [filename, filename, ...])
    def walk(self, top, in_order=True):
        return tf.gfile.Walk(top, in_order)

    def read_file(file, mode = 'rb'):
        with tf.gfile.GFile(file, mode) as fid:
            encode_file = fid.read()
        return encode_file

