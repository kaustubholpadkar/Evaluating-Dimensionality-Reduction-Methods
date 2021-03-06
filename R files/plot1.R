embed_methods <- c("Isomap", "PCA", "tSNE", "UMAP", "kPCA", "LLE", "MDS")
## load test data set
data_set <- loadDataSet("3D S Curve", n = 1000)
## apply dimensionality reduction
data_emb <- lapply(embed_methods, function(x) embed(data_set, x))
names(data_emb) <- embed_methods
## figure \ref{fig:plotexample}a, the data set
# plot(data_set, type = "3vars")
## figures \ref{fig:plotexample}b (Isomap) and \ref{fig:plotexample}d (PCA)
lapply(data_emb, plot, type = "2vars")
## figure \ref{fig:plotexample}c, quality analysis
# plot_R_NX(data_emb)