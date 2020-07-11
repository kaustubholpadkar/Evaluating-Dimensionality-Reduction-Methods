# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:44:16 2019

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
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.metrics.pairwise import pairwise_distances

os.chdir('C:/Users/admin/Desktop/PostG/GRE/Second Take/Applications/Univs/Stony Brook/Fall 19 Courses/DSF/Project')
import Mantel


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train[:, :, :, 0]
x_test = x_test[:, :, :, 0]

# flatten the images
dim = 32*32
x_train = np.resize(x_train, (x_train.shape[0], dim))
x_test = np.resize(x_test, (x_test.shape[0], dim))



## 1. implementing PCA
# Reducing the training set size to be as same in TSNE (5000)
trpc = np.zeros((1, x_train.shape[1]+1))
trpc_y = []
cifLab = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
trainnewpc = np.append(x_train, np.resize(y_train, (50000, 1)), 1)

for i in range(10):
    ind, = np.where(trainnewpc[:, -1]==i)
    sub = ind[:500]
    trpc = np.vstack((trpc, trainnewpc[sub]))

for i in cifLab:
    for j in range(500):
        trpc_y.append(i)

# dropping the top zero row
trpc = trpc[1:,:]
trpc_y = trpc[:, -1]   # may or maynot want
trpc = trpc[:,:-1]


pca = PCA(n_components=2)
time_pca = time.process_time()
Xt_red = pca.fit(trpc).transform(trpc)
time_pca = time.process_time() - time_pca


# visualizing reduced examples
dat = pd.DataFrame({'Col1':Xt_red[:, 0], 'Col2':Xt_red[:, 1], 'label':trpc_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = dat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in cifLab:
    xtext = np.median(dat['Col1'][dat['label'] == i])
    ytext = np.median(dat['Col2'][dat['label'] == i])
    txt = plt.text(xtext, ytext, i, fontsize=15)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



## TSNE

pca50 = PCA(n_components=40)
train50 = pca50.fit(trpc).transform(trpc)
#projection = TSNE().fit_transform(train50)
time_tsne = time.process_time()
projection = TSNE().fit_transform(trpc)
time_tsne = time.process_time() - time_tsne

# trying to visualize clusters
dat = pd.DataFrame({'Col1':projection[:, 0], 'Col2':projection[:, 1], 'label':trpc_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = dat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in cifLab:
    xtext = np.median(dat['Col1'][dat['label'] == i])
    ytext = np.median(dat['Col2'][dat['label'] == i])
    txt = plt.text(xtext, ytext, i, fontsize=15)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



## UMAP
time_umap = time.process_time()
umap50 = umap.UMAP(random_state=42).fit(trpc).transform(trpc)
time_umap = time.process_time() - time_umap

# trying to visualize clusters
dat = pd.DataFrame({'Col1':umap50[:, 0], 'Col2':umap50[:, 1], 'label':trpc_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = dat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in cifLab:
    xtext = np.median(dat['Col1'][dat['label'] == i])
    ytext = np.median(dat['Col2'][dat['label'] == i])
    txt = plt.text(xtext, ytext, i, fontsize=15)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



## LDA
time_lda = time.process_time()
lda50 = LinearDiscriminantAnalysis(n_components=2).fit_transform(trpc, y=trpc_y)
time_lda = time.process_time() - time_lda
# trying to visualize clusters
dat = pd.DataFrame({'Col1':lda50[:, 0], 'Col2':lda50[:, 1], 'label':trpc_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = dat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in cifLab:
    xtext = np.median(dat['Col1'][dat['label'] == i])
    ytext = np.median(dat['Col2'][dat['label'] == i])
    txt = plt.text(xtext, ytext, i, fontsize=15)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



## Autoencoder
ncol = trpc.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(trpc, trpc_y, train_size = 0.9, random_state = np.random.seed(2017))
encoding_dim = 30

input_dim = Input(shape = (ncol, ))


# Encoder Layers
encoded1 = Dense(700, activation = 'relu')(input_dim)
encoded2 = Dense(500, activation = 'relu')(encoded1)
encoded3 = Dense(200, activation = 'relu')(encoded2)
encoded4 = Dense(100, activation = 'relu')(encoded3)
encoded5 = Dense(50, activation = 'relu')(encoded4)
encoded6 = Dense(encoding_dim, activation = 'relu')(encoded5)

# Decoder Layers
decoded1 = Dense(50, activation = 'relu')(encoded6)
decoded2 = Dense(100, activation = 'relu')(decoded1)
decoded3 = Dense(200, activation = 'relu')(decoded2)
decoded4 = Dense(500, activation = 'relu')(decoded3)
decoded5 = Dense(700, activation = 'relu')(decoded4)
decoded6 = Dense(ncol, activation = 'sigmoid')(decoded5)


# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded6)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')



# Train autoencoder
start = time.time()
autoencoder.fit(X_train, X_train, nb_epoch = 10, batch_size = 32, shuffle = False, validation_data = (X_test, X_test))
end = time.time() - start
# Use Encoder level to reduce dimension of data
encoder = Model(inputs = input_dim, outputs = encoded6)
encoded_train = pd.DataFrame(encoder.predict(trpc))

# Now using PCA
pca = PCA(n_components=2)
Xt_red = pca.fit(encoded_train).transform(encoded_train)


# trying to visualize clusters
dat = pd.DataFrame({'Col1':Xt_red[:, 0], 'Col2':Xt_red[:, 1], 'label':trpc_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = dat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in cifLab:
    xtext = np.median(dat['Col1'][dat['label'] == i])
    ytext = np.median(dat['Col2'][dat['label'] == i])
    txt = plt.text(xtext, ytext, i, fontsize=15)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)





## Try Classification with Random Forest
# 1. without dimensionality reduction
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold = model_selection.cross_val_score(model_kfold, trpc, trpc_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
results_kfold = results_kfold.mean()*100

# 2. Using PCA
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_pca = model_selection.cross_val_score(model_kfold, dat.iloc[:, :2], trpc_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_pca.mean()*100.0)) 
results_kfold_pca = results_kfold_pca.mean()*100

# 3. Using t-SNE
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_tsne = model_selection.cross_val_score(model_kfold, dat.iloc[:, :2], trpc_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_tsne.mean()*100.0))
results_kfold_tsne = results_kfold_tsne.mean()*100

# 4. Using UMAP
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_umap = model_selection.cross_val_score(model_kfold, dat.iloc[:, :2], trpc_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_umap.mean()*100.0))
results_kfold_umap = results_kfold_umap.mean()*100


# 7. Using Auto Encoder
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_lda = model_selection.cross_val_score(model_kfold, dat.iloc[:, :2], trpc_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_lda.mean()*100.0))
results_kfold_lda = results_kfold_lda.mean()*100




## Plotting time and accuracy metrics
# 1. Time
objects = ('PCA', 't-SNE', 'UMAP', 'Autoencoder')
y_pos = np.arange(len(objects))
performance = [1.05, 830.18, 38.80, 46.1340]
plt.bar(y_pos, performance, align='center', alpha=0.5)
ax = plt.gca()
# Set x logaritmic
ax.set_yscale('log', basey=2)
plt.xticks(y_pos, objects)
plt.ylabel('Time (seconds)')
plt.title('Running Time - Fashion MNIST')
plt.show()


# 2. Accuracy
objects = ('Original Data', 'PCA', 't-SNE', 'UMAP', 'Autoencoder')
y_pos = np.arange(len(objects))
performance = [87.14, 50.25, 81.79, 74.97, 49.78]
plt.bar(y_pos, performance, align='center', alpha=0.5)
ax = plt.gca()
# Set x logaritmic
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy (%age)')
plt.title('Classification Accuracy - Fashion MNIST')
plt.show()






## Running the Mantel test on the CIFAR dataset
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







## Davies-Bouldin Index
from sklearn.metrics import davies_bouldin_score
davies_bouldin_score(Xt_red, trpc_y)
davies_bouldin_score(projection, trpc_y)
davies_bouldin_score(umap50, trpc_y)
