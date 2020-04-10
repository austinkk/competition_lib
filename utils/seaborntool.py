#!/usr/bin/python
import numpy as np
import seaborn as sns
import pandas as pd


class SeabornTool(object):
    "my seaborn tools"
    def __init__(self):
        pass
    
    def distplot(self, x, hist = True, kde = False, fit = None, norm_hist = True):
        """
        hist : 是否绘制（标准化）直方图。
        kde : 是否绘制高斯核密度估计图
        fit : 输入是一个带有fit的对象，返回一个元祖 如 fit = stats.gamma
        norm_hist : 如果为True，则直方图的高度显示密度而不是计数。如果绘制KDE图或拟合密度，则默认为True。
        """
        sns.distplot(x, hist = hist, kde=kde, fit = fit, norm_hist = norm_hist)

    def jointplot(self, x_name, y_name, df, kind = "scatter"):
        """
        散点图
        kind = “scatter” | “reg” | “resid” | “kde” | “hex” 数据量较大用"hex"
        """
        sns.jointplot(x=x_name, y=y_name, data=df, kind = kind)

    def pairplot(self, df, label_name = None):
        """
        多特征对比
        """
        sns.pairplot(df, hue = label_name)
        
    def lmplot(self, x_name, y_name, df, robust = True,  order = 1, row = None, col = None, label_name = None):
        """
        拟合数据
        robust : 是否忽略异常值
        order : 多项式拟合系数
        """
        sns.lmplot(x=x_name, y=y_name, data=df, robust=robust, order = order, row = row, col = col, hue = label_name)

    def swarmplot(self, x_name, y_name, df, label_name = None):
        """
        单特征分析
        """
        sns.swarmplot(x=x_name, y=y_name, hue=label_name,data=df)
        sns.violinplot(x=x_name, y=y_name hue=label_name, data=df, split=True)

    def boxplot(self, x_name, y_name, df, label_name = None):
        """
        盒图
        """
        sns.boxplot(x=x_name, y=y_name, hue=label_name, data=df)

    def heatmap(self, data, vmin, vmax, center):
        """
        vmin 指定小于一个数值是一个颜色，vmax 指定大于一个数值是一个颜色
        center 指定两边颜色变浅
        """
        sns.heatmap(data,vmin=vmin, vmax=vmax,center=center,annot=True,fmt="d",linewidths=.5)
