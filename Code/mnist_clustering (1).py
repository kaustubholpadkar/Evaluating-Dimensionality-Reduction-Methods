# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:28:04 2019

@author: DHYANI
"""

# dimensionality reduction of MNIST

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


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
for i in range(10):
    xtext = np.median(dat['Col1'][dat['label'] == i])
    ytext = np.median(dat['Col2'][dat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)

# saving the picture
pic = fig.get_figure()
pic.savefig('foo.png')
plt.scatter(Xt_red[:, 0], Xt_red[:, 1])
# Looks like PCA has piled up at one place but couldn't separate the clusters

# Using clustering HDBSCAN
df = pd.DataFrame({'Col1':Xt_red[:, 0], 'Col2':Xt_red[:, 1]})
clusterer = hdbscan.HDBSCAN()
clusterer.fit(df)
clusterer.labels_
# very large number of clusters using hdbscan
clusterer.labels_.max()
# 2329 clusters, when actually there are just 10 clusters



## 2. Implementing t-SNE

# Before using t-SNE directly, reduce the dimensions to 50 using PCA to speed up computation
# for t-SNE
# Also reduce the training examples - taking 500 examples each
tr = np.zeros((1, x_train.shape[1]+1))
trainnew = np.append(x_train, np.resize(y_train, (60000, 1)), 1)

for i in range(10):
    ind, = np.where(trainnew[:, -1]==i)
    sub = ind[:500]
    tr = np.vstack((tr, trainnew[sub]))

# dropping the top zero row
tr = tr[1:,:]
tr_y = tr[:, -1]
tr = tr[:,:-1]

pca50 = PCA(n_components=40)
train50 = pca50.fit(tr).transform(tr)
#projection = TSNE().fit_transform(train50)
time_tsne = time.process_time()
projection = TSNE().fit_transform(trpc)
time_tsne = time.process_time() - time_tsne

# trying to visualize clusters
tsnedat = pd.DataFrame({'Col1':projection[:, 0], 'Col2':projection[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = tsnedat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(tsnedat['Col1'][tsnedat['label'] == i])
    ytext = np.median(tsnedat['Col2'][tsnedat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)




## 3. Implementing UMAP
# On the same tr and tr_y reduced dataset
time_umap = time.process_time()
umap50 = umap.UMAP(random_state=42).fit(trpc).transform(trpc)
time_umap = time.process_time() - time_umap

# trying to visualize clusters
umapdat = pd.DataFrame({'Col1':umap50[:, 0], 'Col2':umap50[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = umapdat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(umapdat['Col1'][umapdat['label'] == i])
    ytext = np.median(umapdat['Col2'][umapdat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



## 4. Implementing LDA (Linear Discriminant Analysis)
time_lda = time.process_time()
lda50 = LinearDiscriminantAnalysis(n_components=2).fit_transform(tr, y=tr_y)
time_lda = time.process_time() - time_lda

# trying to visualize clusters
ldadat = pd.DataFrame({'Col1':lda50[:, 0], 'Col2':lda50[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = ldadat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(ldadat['Col1'][ldadat['label'] == i])
    ytext = np.median(ldadat['Col2'][ldadat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)




## 5. SVD
time_svd = time.process_time()
svd50 = TruncatedSVD(n_components=2).fit_transform(tr)
time_svd = time.process_time() - time_svd

# trying to visualize clusters
svddat = pd.DataFrame({'Col1':svd50[:, 0], 'Col2':svd50[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = svddat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(svddat['Col1'][svddat['label'] == i])
    ytext = np.median(svddat['Col2'][svddat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



## Try Classification with Random Forest
# 1. without dimensionality reduction
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold = model_selection.cross_val_score(model_kfold, tr, tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
results_kfold = results_kfold.mean()*100

# 2. Using PCA
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_pca = model_selection.cross_val_score(model_kfold, dat.iloc[:, :2], tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_pca.mean()*100.0)) 
results_kfold_pca = results_kfold_pca.mean()*100

# 3. Using t-SNE
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_tsne = model_selection.cross_val_score(model_kfold, tsnedat.iloc[:, :2], tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_tsne.mean()*100.0))
results_kfold_tsne = results_kfold_tsne.mean()*100

# 4. Using UMAP
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_umap = model_selection.cross_val_score(model_kfold, umapdat.iloc[:, :2], tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_umap.mean()*100.0))
results_kfold_umap = results_kfold_umap.mean()*100

# 5. Using SVD
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_svd = model_selection.cross_val_score(model_kfold, svddat.iloc[:, :2], tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_svd.mean()*100.0))
results_kfold_svd = results_kfold_svd.mean()*100

# 6. Using LDA
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_lda = model_selection.cross_val_score(model_kfold, ldadat.iloc[:, :2], tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_lda.mean()*100.0))
results_kfold_lda = results_kfold_lda.mean()*100



## Plotting time and accuracy metrics
# 1. Time
objects = ('PCA', 't-SNE', 'UMAP', 'LDA', 'SVD')
y_pos = np.arange(len(objects))
performance = [time_pca, time_tsne, time_umap, time_lda, time_svd]
plt.bar(y_pos, performance, align='center', alpha=0.5)
ax = plt.gca()
# Set x logaritmic
ax.set_yscale('log', basey=2)
plt.xticks(y_pos, objects)
plt.ylabel('Time (seconds)')
plt.title('Running Time - MNIST')
plt.show()


# 2. Accuracy
objects = ('Original Data', 'PCA', 't-SNE', 'UMAP', 'LDA', 'SVD')
y_pos = np.arange(len(objects))
performance = [results_kfold, results_kfold_pca, results_kfold_tsne, results_kfold_umap, results_kfold_lda, results_kfold_svd]
plt.bar(y_pos, performance, align='center', alpha=0.5)
ax = plt.gca()
# Set x logaritmic
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy (%age)')
plt.title('Classification Accuracy - MNIST')
plt.show()





# ---------------- Project Progress -------------------------
# Autoencoders for DR
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

ncol = tr.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(tr, tr_y, train_size = 0.9, random_state = np.random.seed(2017))
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
encoded_train = pd.DataFrame(encoder.predict(tr))

# Now using PCA to reduce 3 dimensions into 2... Autoencoders is not giving good output
# for 2 dimensions, but only for 3 dimensions
pca = PCA(n_components=2)
Xt_red = pca.fit(encoded_train).transform(encoded_train)
# trying to visualize clusters
umapdat = pd.DataFrame({'Col1':Xt_red[:, 0], 'Col2':Xt_red[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = umapdat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(umapdat['Col1'][umapdat['label'] == i])
    ytext = np.median(umapdat['Col2'][umapdat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



# Trying UMAP instead of PCA
umap50 = umap.UMAP(random_state=42).fit(encoded_train).transform(encoded_train)
# Visualize
umapdat = pd.DataFrame({'Col1':umap50[:, 0], 'Col2':umap50[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = umapdat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(umapdat['Col1'][umapdat['label'] == i])
    ytext = np.median(umapdat['Col2'][umapdat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



# Trying T-SNE instead of UMAP
projection = TSNE().fit_transform(encoded_train)
# Visualize
umapdat = pd.DataFrame({'Col1':projection[:, 0], 'Col2':projection[:, 1], 'label':tr_y})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = umapdat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(umapdat['Col1'][umapdat['label'] == i])
    ytext = np.median(umapdat['Col2'][umapdat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)


# Classification accuracy:
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_pca = model_selection.cross_val_score(model_kfold, Xt_red, tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_pca.mean()*100.0)) 
results_kfold_pca = results_kfold_pca.mean()*100



## Now autoencoder for Fashion MNIST
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

(fX_train, fy_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
fX_train = np.reshape(fX_train, newshape=(-1, 28*28))
tr = np.zeros((1, fX_train.shape[1]+1))
trainnew = np.append(fX_train, np.resize(fy_train, (60000, 1)), 1)

for i in range(10):
    ind, = np.where(trainnew[:, -1]==i)
    sub = ind[:500]
    tr = np.vstack((tr, trainnew[sub]))

# dropping the top zero row
tr = tr[1:,:]
tr_y = tr[:, -1]
tr = tr[:,:-1]


labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Same as before

ncol = tr.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(tr, tr_y, train_size = 0.9, random_state = np.random.seed(2017))
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
autoencoder.fit(X_train, X_train, nb_epoch = 5, batch_size = 32, shuffle = False, validation_data = (X_test, X_test))
end = time.time() - start
# Use Encoder level to reduce dimension of data
encoder = Model(inputs = input_dim, outputs = encoded6)
encoded_train = pd.DataFrame(encoder.predict(fX_train))


## >>> Again all those PCA, UMAP and TSNE on the reduced datasets and visualization
# Now using PCA to reduce 3 dimensions into 2... Autoencoders is not giving good output
# for 2 dimensions, but only for 3 dimensions
pca = PCA(n_components=2)
Xt_red = pca.fit(tr).transform(tr)
# trying to visualize clusters
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)], label=labels)
    ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Label')

    legs = (plt.scatter(np.random.random(10), np.random.random(10), marker='o', color=palette[i]) for i in range(10))

    ax.legend(legs,
           labels,
           scatterpoints=1,
           ncol=3,
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plt.xlim(25, 25)
    # plt.ylim(25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, labels[i], fontsize=14)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

pca_df = pd.DataFrame({'Col1':Xt_red[:, 0], 'Col2':Xt_red[:, 1]})
fashion_scatter(pca_df[['Col1','Col2']].values,fy_train)


# Trying UMAP instead of PCA
umap50 = umap.UMAP(random_state=42).fit(encoded_train).transform(encoded_train)
# Visualize
umap_df = pd.DataFrame({'Col1':umap50[:, 0], 'Col2':umap50[:, 1]})
fashion_scatter(umap_df[['Col1','Col2']].values,fy_train)



# Trying T-SNE instead of UMAP
projection = TSNE().fit_transform(encoded_train)
# Visualize
umapdat = pd.DataFrame({'Col1':projection[:, 0], 'Col2':projection[:, 1], 'label':fy_train})
color_palette = sns.color_palette('deep', 10)
plt.figure(figsize=(9,7))
fig = sns.scatterplot(data = umapdat, x='Col1', y='Col2', hue='label', legend='full', palette=color_palette)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
txts = []
for i in range(10):
    xtext = np.median(umapdat['Col1'][umapdat['label'] == i])
    ytext = np.median(umapdat['Col2'][umapdat['label'] == i])
    txt = plt.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)



# Classification accuracy:
skfold = model_selection.StratifiedKFold(n_splits=3, random_state=100)
model_kfold = RandomForestClassifier(n_estimators=100, random_state=10)
results_kfold_pca = model_selection.cross_val_score(model_kfold, Xt_red, tr_y, cv=skfold)
print("Accuracy: %.2f%%" % (results_kfold_pca.mean()*100.0)) 
results_kfold_pca = results_kfold_pca.mean()*100






## Mantel Test on MNIST
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

