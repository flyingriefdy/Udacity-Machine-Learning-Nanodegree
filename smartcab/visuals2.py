# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:12:51 2017

@author: hazie
"""

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################
#

###########################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import ast
import pylab as pl
import matplotlib as mpl


def calculate_safety(data):
	""" Calculates the safety rating of the smartcab during testing. """

	good_ratio = data['good_actions'].sum() * 1.0 / \
	(data['initial_deadline'] - data['final_deadline']).sum()

	if good_ratio == 1: # Perfect driving
		return (0, "green")
	else: # Imperfect driving
		if data['actions'].apply(lambda x: ast.literal_eval(x)[4]).sum() > 0: # Major accident
			return (5, "red")
		elif data['actions'].apply(lambda x: ast.literal_eval(x)[3]).sum() > 0: # Minor accident
			return (4, "#EEC700")
		elif data['actions'].apply(lambda x: ast.literal_eval(x)[2]).sum() > 0: # Major violation
			return (3, "#EEC700")
		else: # Minor violation
			minor = data['actions'].apply(lambda x: ast.literal_eval(x)[1]).sum()
			if minor >= len(data)/2: # Minor violation in at least half of the trials
				return (2, "green")
			else:
				return (1, "green")


def calculate_reliability(data):
	""" Calculates the reliability rating of the smartcab during testing. """

	success_ratio = data['success'].sum() * 1.0 / len(data)

	if success_ratio == 1: # Always meets deadline
		return (0, "green")
	else:
		if success_ratio >= 0.90:
			return (1, "green")
		elif success_ratio >= 0.80:
			return (2, "green")
		elif success_ratio >= 0.70:
			return (3, "#EEC700")
		elif success_ratio >= 0.60:
			return (4, "#EEC700")
		else:
			return (5, "red")


def plot_safety_reliability():
    ''' Plot safety and reliability rating of different csv'''
    # Create dataframe to store epsilon, alpha, reliability, safety
    df = pd.DataFrame()  
    # Counter for index
    counter = 0
    # Iteration to store reliability and safety into dataframe
    for i in range(1, 11):
        for j in range(1,11):
            # Check if file exists
            if os.path.isfile("logs/sim_improved-learning-epsilon-{}-alpha-{}.csv".format(i/10.0,j/10.0)) == True:
                data = pd.read_csv("logs/sim_improved-learning-epsilon-{}-alpha-{}.csv".format(i/10.0,j/10.0))
                # Create additional features
                data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
                testing_data = data[data['testing'] ==True]
                if len(testing_data) > 0:
                    reliability = calculate_reliability(testing_data) # Getting reliability value
                    safety = calculate_safety(testing_data) # Getting safety value
                    df = df.append(pd.DataFrame({"epsilon": i/10.0, "alpha": j/10.0, "reliability": reliability[0], "safety": safety[0]}, index = [counter]))
                    counter += 1
                    print df
            else:
                print"logs/sim_improved-learning-epsilon-{}-alpha-{}.csv does not exist".format(i/10.0,j/10.0)

    # Define the data
    X = df.iloc[:,0].values
    y = df.iloc[:,1].values
    tag_reliability = df.iloc[:,2].values
    tag_safety = df.iloc[:,3].values

    # Number of labels
    N = 7
    # Setup plot
    plt.figure(1)
    ax = plt.subplot(211)

    # Define the colormap
    cmap = plt.cm.jet
    # Extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # Create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # Define the bins and normalise
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    tick_labels = ['A+','A','B','C','D','E','F']

    # Make the scatter
    scat = plt.scatter(X, y, c = tag_reliability, s = 100, cmap = cmap, norm = norm)

    # Create the colorbar
    cb = plt.colorbar(scat, spacing = 'proportional', ticks = bounds)
    cb.set_ticklabels(ticklabels= tick_labels)
    cb.set_label('Reliability color bar')
    ax.set_title('Reliability ratings mapping')
     
    # Label axis
    plt.ylabel('epsilon')
    plt.xlabel('alpha')
    
    # Setup plot
    ax2 = plt.subplot(212)
    # Make the scatter    
    scat2 = plt.scatter(X, y, c = tag_safety, s = 100, cmap = cmap, norm = norm)
    # Create the colorbar
    cb2 = plt.colorbar(scat2, spacing = 'proportional', ticks = bounds)
    cb2.set_ticklabels(ticklabels= tick_labels)
    cb2.set_label('Safety color bar')
    ax2.set_title('Safety ratings mapping')
    
    # Label axis
    plt.ylabel('epsilon')
    plt.xlabel('alpha')
    
    plt.show()
    
 
            
if __name__ == '__main__':
    plot_safety_reliability()            
                   
       