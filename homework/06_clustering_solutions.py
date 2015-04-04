'''
CLUSTER ANALYSIS on Countries
How do we implement a k-means clustering algorithm?

scikit-learn KMeans documentation for reference:
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the data
d = pd.read_csv('../data/UNdata.csv')

cols = ['GDPperCapita', 'infantMortality', 'lifeFemale', 'lifeMale']
np.random.seed(0)

# Run KMeans with k = 3
# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality
est = KMeans(n_clusters=3, init='random')
est.fit(d[cols])
y_kmeans = est.predict(d[cols])

# Print out the countries present within each cluster. Do you notice any general trend?
d['cluster'] = y_kmeans

d.sort_index(by=['cluster', 'region', 'country'])[['cluster', 'region', 'country']]

d.loc[d['cluster'] == 0, ['region', 'country']].sort_index(by='region')
d.loc[d['cluster'] == 1, ['region', 'country']].sort_index(by='region')
d.loc[d['cluster'] == 2, ['region', 'country']].sort_index(by='region')


# Print out the properties of each cluster. What are the most striking differences?
d.groupby('cluster').lifeMale.mean()
d.groupby('cluster')[cols].mean()

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)
colors = np.array(['#FF0054','#FBD039','#23C2BC'])

# Ex. Scatter Plot Grid
plt.figure(figsize=(15, 7))
plt.suptitle('Scatter Plot Grid (Unscaled)',  fontsize=14)
# Left most
plt.subplot(131)
plt.scatter(d['infantMortality'], d['GDPperCapita'], c = colors[d['cluster']])
plt.ylabel('GDPperCapita')
plt.xlabel('infantMortality')

# Middle
plt.subplot(132)
plt.scatter(d['lifeFemale'], d['GDPperCapita'], c = colors[d['cluster']])
plt.xlabel('lifeFemale')

# Right most
plt.subplot(133)
plt.scatter(d['lifeMale'], d['GDPperCapita'], c = colors[d['cluster']])
plt.xlabel('lifeMale')

# Advanced: Re-run the cluster analysis after centering and scaling all four variables 
cols = ['GDPperCapita', 'infantMortality', 'lifeFemale', 'lifeMale']
dcs = (d[cols] - d[cols].mean(axis=0)) / d[cols].std(axis=0)

# BONUS: See link below for a thorough discussion on "Should I standardize my data?"
# http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html

# Advanced: How do the results change after they are centered and scaled? Why is this?
est = KMeans(n_clusters=3, init='random')
est.fit(dcs)
y_kmeans = est.predict(dcs)
dcs['cluster'] = y_kmeans

# Plotting
plt.figure(figsize=(15, 7))
plt.suptitle('Scatter Plot Grid (Scaled)',  fontsize=14)
plt.subplot(131)
plt.scatter(dcs['infantMortality'], dcs['GDPperCapita'], c = colors[dcs['cluster']])
plt.ylabel('GDPperCapita')
plt.xlabel('infantMortality')
plt.subplot(132)
plt.scatter(dcs['lifeFemale'], dcs['GDPperCapita'], c = colors[dcs['cluster']])
plt.xlabel('lifeFemale')
plt.subplot(133)
plt.scatter(dcs['lifeMale'], dcs['GDPperCapita'], c = colors[dcs['cluster']])
plt.xlabel('lifeMale')

'''
The scale on the GDPperCapita dimension was dominating the 
K-means objective function and thus the other measures were de-prioritized 
in the clustering routine.
'''

# BONUS: Creating the other plots
# 3d plot
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure(figsize=(10, 9)), rect=[.01, 0, 0.95, 1], elev=30, azim=134)
ax.scatter(dcs['infantMortality'], 
           dcs['GDPperCapita'], 
           dcs['lifeFemale'], 
            c = colors[y_kmeans], s=120)
ax.set_xlabel('infantMortality')
ax.set_ylabel('GDPperCapita')
ax.set_zlabel('lifeFemale')
# Modified from the example here: 
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html

# Parallel Coordinates plot
plt.figure()
plt.suptitle('Parallel Coordinates', fontsize=15)
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(data=dcs, class_column='cluster', 
                     colors=('#FF0054', '#FBD039', '#23C2BC'))
