# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class FDM:
    def __init__(self, H, W):
        self.H_steps = np.linspace(0., 1., H, endpoint=False)
        self.W_steps = np.linspace(0., 1., W, endpoint=False)
        self.H = H
        self.W = W
        self.Size = np.dstack(np.meshgrid(self.H_steps, self.W_steps)).reshape((-1, 2))
    
    def make_heatmap(self, gaze_point_x, gaze_point_y):
        gp = np.vstack((np.array(gaze_point_x), np.array(gaze_point_y))).astype(np.float64)
        kde = sm.nonparametric.KDEMultivariate(data=[gp[0,:], gp[1,:]], var_type='cc', bw=[0.05, 0.05])
        self.Map = (kde.pdf(self.Size) * gp.shape[1]).reshape((self.H, self.W))
    
    def draw_heatmap(self):
        plt.imshow(self.Map)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.show()
        plt.close()
    
#gp = pd.read_csv("./sample.csv")

#fdm = FDM(100,100)
#fdm.make_heatmap(gp['GazePoint_X'], gp['GazePoint_Y'])
#fdm.draw_heatmap()