'''
CLUSTER ANALYSIS ON COUNTRIES
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the data
d = pd.read_csv('../data/UNdata.csv')
np.random.seed(0)


# Run KMeans with k = 3
# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)

# Print out the countries present within each cluster. Do you notice any general trend?

# Print out the properties of each cluster. What are the most striking differences?
             
# Advanced: Re-run the cluster analysis after centering and scaling all four variables 
             
# Advanced: How do the results change after they are centered and scaled? Why is this?