library("dimRed")

embed_methods <- c("AutoEncoder", "DiffusionMaps", "DRR", "FastICA", "KamadaKawai", "DrL", "FruchtermanReingold", "HLLE", "Isomap", "kPCA", "PCA_L1", "LaplacianEigenmaps", "LLE", "MDS", "nMDS", "PCA", "tSNE", "UMAP")

quality_methods <- c("Q_local", "Q_global", "mean_R_NX", "AUC_lnK_R_NX", "total_correlation", "cophenetic_correlation", "distance_correlation", "reconstruction_rmse")


scurve <- loadDataSet("3D S Curve", n = 2000)

quality_results <- matrix(NA, length(embed_methods), length(quality_methods), dimnames = list(embed_methods, quality_methods))

embedded_data <- list()

for (e in embed_methods) {
embedded_data[[e]] <- embed(scurve, e)
print("here")
for (q in quality_methods)
try(quality_results[e, q] <- quality(embedded_data[[e]], q))
}
