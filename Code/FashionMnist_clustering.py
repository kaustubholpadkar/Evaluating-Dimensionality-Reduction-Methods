# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:46:43 2019

@author: DHYANI
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import hdbscan
import umap
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics.pairwise import pairwise_distances
import os

os.chdir('C:/Users/admin/Desktop/PostG/GRE/Second Take/Applications/Univs/Stony Brook/Fall 19 Courses/DSF/Project')
import Mantel

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# flatten the images
dim = 28*28
x_train = np.resize(x_train, (x_train.shape[0], dim))
x_test = np.resize(x_test, (x_test.shape[0], dim))


## 1. implementing PCA
# Reducing the training set size to be as same in TSNE (5000)
trpc = np.zeros((1, x_train.shape[1]+1))
trainnewpc = np.append(x_train, np.resize(y_train, (60000, 1)), 1)

for i in range(10):
    ind, = np.where(trainnewpc[:, -1]==i)
    sub = ind[:500]
    trpc = np.vstack((trpc, trainnewpc[sub]))

# dropping the top zero row
trpc = trpc[1:,:]
trpc_y = trpc[:, -1]
trpc = trpc[:,:-1]

# 1. PCA
pca = PCA(n_components=2)
time_pca = time.process_time()
Xt_red = pca.fit(trpc).transform(trpc)
time_pca = time.process_time() - time_pca

# 2. TSNE
time_tsne = time.process_time()
projection = TSNE().fit_transform(trpc)
time_tsne = time.process_time() - time_tsne

# 3. UMAP
umap50 = umap.UMAP(random_state=42).fit(trpc).transform(trpc)




## Mantel Test on Fashion MNIST dataset
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial
from scipy.spatial.distance import pdist, squareform
# First create pairwise distance matrix of the dataset
distance_mat = pairwise_distances(trpc, metric='euclidean')
# now compute the distance matrix on the reduced dataset
distance_pca = pdist(umap50, metric='euclidean')
distance_pca = squareform(distance_pca)
spatial.distance.is_valid_dm(distance_pca)
# check if symmetric
(distance_pca.transpose() == distance_pca).all()

# now can run the mantel test on both these distance matrices
Mantel.test(distance_mat, distance_pca, perms=100, method='pearson', tail='upper')








## Visualization for all the Mantel Tests
# Create a dataframe containing all the values we obtained
corr = [0.55282730781341, 0.4690514075130153, 0.3305476387379797,
        0.8750741193546719, 0.6727758713508317, 0.6201096283284083,
        0.8250056707357839, 0.7209156625017631, 0.7272945537430893]
dr = ['PCA', 'T-SNE', 'UMAP']*3
dataset = ['MNIST']*3 + ['FashionMNIST']*3 + ['CIFAR']*3
dframe = pd.DataFrame({'Correlation':corr, 'DR Methods':dr, 'Dataset':dataset})
sns.lineplot(data=dframe, x='DR Methods', y='Correlation', hue='Dataset', legend='full')






## Davies-Bouldin Index
from sklearn.metrics import davies_bouldin_score
davies_bouldin_score(Xt_red, trpc_y)
davies_bouldin_score(projection, trpc_y)
davies_bouldin_score(umap50, trpc_y)



## Visualizing Davies-Bouldin Index
ind = [7.043016845414023, 1.4132924812601786, 1.6275283521787358,
        3.7046187659113476, 2.211231690595157, 2.00341490812914,
        14.641700450672857, 18.04887396913739, 14.357904691642428]
dr = ['PCA', 'T-SNE', 'UMAP']*3
dataset = ['MNIST']*3 + ['FashionMNIST']*3 + ['CIFAR']*3
dframe = pd.DataFrame({'Davies-Bouldin Index':ind, 'DR Methods':dr, 'Dataset':dataset})
sns.lineplot(data=dframe, x='DR Methods', y='Davies-Bouldin Index', hue='Dataset', legend='full')


