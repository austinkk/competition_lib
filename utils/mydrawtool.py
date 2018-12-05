#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

class MyDrawTool(object):
    "my tools for drawing picture"
    def __init__(self, figsize = (9,6), look = 'ggplot'):
        self.cols = ['r','y','g','c','b','m','k','w']
        self.figsize = figsize
        style.use(look)
        
    def pie(self, data, labels, title = 'Pie Plot'):
        plt.figure(figsize=self.figsize)
        l = len(data)
        plt.pie(data,
                labels=labels,
                colors=self.cols[:l],
                startangle=90,
                shadow= True,
                #explode=[0,0,0,0],
                autopct='%1.1f%%')
    
    def area(self, x_data, y_data, labels):
        plt.figure(figsize=self.figsize)
        l = len(y_data)
        for i in range(l):
            plt.plot([],[],color=self.cols[i], label=labels[i], linewidth=5)
        plt.stackplot(x_data, *y_data, colors=self.cols[:l])
        
    def point(self, x_data, y_data, labels):
        plt.figure(figsize=self.figsize)
        l = len(y_data)
        for i in range(l):
            plt.scatter(x_data[i],y_data[i], label=labels[i],color=self.cols[i])
    
    def hist(self, data, bins):
        plt.figure(figsize=self.figsize)
        plt.hist(data, bins, histtype='bar', color='b', rwidth=0.8)
    
    def bar(self, data, labels,xticks, width = 0.5):
        plt.figure(figsize=self.figsize)
        l = len(data)
        data_l = len(data[0])
        x_pos = [[] for i in range(l)]
        s_pos = 0.0
        for i in range(data_l):
            for j in range(l):
                x_pos[j].append(s_pos)
                s_pos += width
            s_pos += width

        for i in range(l):
            plt.bar(x_pos[i], data[i], label=labels[i], color=self.cols[i], width=width)
        plt.xticks([(x_pos[0][i] + x_pos[-1][i])/2.0  for i in range(data_l)], xticks, rotation=40)

    
    def line(self, x_data, y_data, labels):
        plt.figure(figsize=self.figsize)
        l = len(y_data)
        for i in range(l):
            plt.plot(x_data[i],y_data[i], label=labels[i],color=self.cols[i], linewidth=2)
    
    def addGrid(self, linestyle = '--', color = 'k', linewidth = 0.5):
        plt.grid(True,linestyle = linestyle,color=color, linewidth =linewidth)
    
    def axis(self, xlim, ylim, xticks, yticks, xticklabels, yticklabels):
        plt.xlim(xlim)  # x轴边界
        plt.ylim(ylim)  # y轴边界
        plt.xticks(xticks)  # 设置x刻度
        plt.yticks(yticks)  # 设置y刻度
        fig.set_xticklabels(xticklabels)  # x轴刻度标签
        fig.set_yticklabels(yticklabels)  # y轴刻度标签
        
    def draw(self, x_l = 'x', y_l = 'y', title = 'Pic'):
        plt.xlabel(x_l)
        plt.ylabel(y_l)
        plt.title(title)
        plt.legend()
        plt.show()

    def addText(self, x, y, text, fontsize = 12):
        plt.text(x,y,text,fontsize=fontsize)
        
    # TODO: split data into bins    
    def getHistBins(self, data):
        pass
