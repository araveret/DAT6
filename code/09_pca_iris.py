"""

Principal Component Analysis applied to the Iris dataset.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

# Load in the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names




#############################
### PCA with 2 components  ##
#############################


pca = decomposition.PCA(n_components=2)
X_r = pca.fit_transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA(2 components) of IRIS dataset')




#############################
### PCA with 3 components  ##
#############################


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)




plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X_3 = pca.transform(X)


# making a pretty 3D graph
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X_3[y == label, 0].mean(),
              X_3[y == label, 1].mean() + 1.5,
              X_3[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y_1 = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X_3[:, 0], X_3[:, 1], X_3[:, 2], c=y_1, cmap=plt.cm.spectral)

x_surf = [X_3[:, 0].min(), X_3[:, 0].max(),
          X_3[:, 0].min(), X_3[:, 0].max()]
y_surf = [X_3[:, 0].max(), X_3[:, 0].max(),
          X_3[:, 0].min(), X_3[:, 0].min()]
x_surf = np.array(x_surf)
y_surf = np.array(y_surf)
v0 = pca.transform(pca.components_[0])
v0 /= v0[-1]
v1 = pca.transform(pca.components_[1])
v1 /= v1[-1]

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


#############################
### choosing components  ####
#############################



pca = decomposition.PCA(n_components=4)
X_r = pca.fit_transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.cla()
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance explained by each principal component')
plt.ylabel(' % Variance Explained')
plt.xlabel('Principal component')

