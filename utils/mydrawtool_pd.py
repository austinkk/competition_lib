#!/usr/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class MyDrawToolPD(object):
    "for series and dataframe"
    def __init__(self, figsize = (9,6)):
        self.cols = ['r','y','g','c','b','m','k','w']
        self.linestyle = ['-', '--', '-.', ':']
        self.marker = ['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
        self.figsize = figsize
    
    def getRandomStyle(self, num):
        linestyle = random.sample(self.linestyle,num)
        marker = random.sample(self.marker,num)
        col = random.sample(self.cols,num)
        return [linestyle[i]+marker[i]+col[i] for i in range(num)]
    
    def getRandomColor(self, num):
        return random.sample(self.cols,num)
        
    def line_series(self, series, label, title = 'Pic', looklike = 'default', alpha = 0.4):
        if looklike == 'default':
            looklike = self.getRandomStyle(1)
        #ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
        series.plot(kind='line',
                    label = label,
                    style = looklike,
                    alpha = alpha,
                    use_index = True,
                    rot = 45,
                    grid = False,
                    figsize = self.figsize,
                    title = title,
                    legend = True)

    def line_df(self, df, title = 'Pic', looklike = 'default', alpha = 0.4, subplots = False):
        if looklike == 'default':
            looklike = self.getRandomStyle(len(df.columns))
        #df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD')).cumsum()
        df.plot(kind='line',
                style = looklike,
                alpha = alpha,
                use_index = True,
                rot = 45,
                grid = False,
                figsize = self.figsize,
                title = title,
                subplots = subplots,
                legend = True)
    
    def bar_series(self, series, title = 'Pic', looklike = 'default', alpha = 0.4):
        if looklike == 'default':
            looklike = self.getRandomStyle(len(series))
        #ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
        series.plot(kind='bar',
                    style = looklike,
                    alpha = alpha,
                    use_index = True,
                    rot = 45,
                    grid = False,
                    figsize = self.figsize,
                    title = title,
                    legend = False)
    
    def bar_df(self, df, title = 'Pic', looklike = 'default', alpha = 0.4):
        if looklike == 'default':
            looklike = self.getRandomStyle(len(df.columns))
        #df = pd.DataFrame(np.random.rand(10,3), columns=['a','b','c'])
        df.plot(kind='bar',
                style = looklike,
                alpha = alpha,
                use_index = True,
                rot = 45,
                grid = False,
                figsize = self.figsize,
                title = title,
                legend = False)

    def bar_stacked_df(self, df, title = 'Pic', looklike = 'default', alpha = 0.4):
        if looklike == 'default':
            looklike = self.getRandomStyle(len(df.columns))
        #df = pd.DataFrame(np.random.rand(10,3), columns=['a','b','c'])
        df.plot(kind='bar',
                style = looklike,
                alpha = alpha,
                use_index = True,
                rot = 45,
                grid = False,
                figsize = self.figsize,
                title = title,
                legend = False,
                stacked = True)
    
    def area_df(self, df, title = 'Pic', looklike = 'default', alpha = 0.4):
        if looklike == 'default':
            looklike = self.getRandomStyle(len(df.columns))
        df.plot.area(style = looklike,
                     alpha = alpha,
                     use_index = True,
                     rot = 45,
                     grid = False,
                     figsize = self.figsize,
                     title = title,
                     legend = True)
        
    def pie_series(self, series, title = 'Pic', looklike = 'default', alpha = 0.4):
        if looklike == 'default':
            looklike = self.getRandomColor(len(series))
        #ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
        plt.axis('equal')  # 保证长宽相等
        plt.pie(series,
                #explode = [0.1,0,0,0],
                labels = series.index,
                colors=looklike,
                autopct='%.2f%%',
                pctdistance=0.6,
                labeldistance = 1.2,
                shadow = True,
                startangle=0,
                radius=1.5,
                frame=False)

    def hist_series(self, series, bins, density = False, alpha = 0.4):
        series.hist(bins = bins,
               histtype = 'bar',
               align = 'mid',
               orientation = 'vertical',
               alpha=0.5,
               density = density)

        # 密度图
        if density:
            series.plot(kind='kde',style='k--')
    
    def scatter_matrix(self, df, alpha = 0.5):
        pd.plotting.scatter_matrix(df,figsize=self.figsize,
                                   marker = 'o',
                                   diagonal='kde',
                                   alpha = alpha,
                                   range_padding=0.5)

    def corrcoef_df(self, df):
        cov = np.corrcoef(df)
        img = plt.matshow(cov,cmap=plt.cm.bwr)
        plt.colorbar(img, ticks=[-1,0,1])

    def box_df(self, df):
        color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
        df.plot.box(color = color)

    def axis(self, xlim, ylim, xticks, yticks, xticklabels, yticklabels):
        plt.xlim(xlim)  # x轴边界
        plt.ylim(ylim)  # y轴边界
        plt.xticks(xticks)  # 设置x刻度
        plt.yticks(yticks)  # 设置y刻度
        fig.set_xticklabels(xticklabels)  # x轴刻度标签
        fig.set_yticklabels(yticklabels)  # y轴刻度标签    

    def draw(self):
        plt.legend()
        plt.show()

    def addText(self, x, y, text, fontsize = 12):
        plt.text(x,y,text,fontsize=fontsize)
    
    def addGrid(self, linestyle = '--', color = 'k', linewidth = 0.5):
        plt.grid(True,linestyle = linestyle,color=color, linewidth =linewidth)
        
    # TODO: split data into bins    
    def getHistBins(self, data):
        pass