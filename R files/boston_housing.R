library("dimRed")

embed_methods <- c("AutoEncoder", "DiffusionMaps", "DRR", "FastICA", "KamadaKawai", "DrL", "FruchtermanReingold", "HLLE", "Isomap", "kPCA", "PCA_L1", "LaplacianEigenmaps", "LLE", "MDS", "nMDS", "PCA", "tSNE", "UMAP")

quality_methods <- c("Q_local", "Q_global", "mean_R_NX", "AUC_lnK_R_NX", "total_correlation", "cophenetic_correlation", "distance_correlation", "reconstruction_rmse")

data(BostonHousing)

quality_results <- matrix(NA, length(embed_methods), length(quality_methods), dimnames = list(embed_methods, quality_methods))

embedded_data <- list()

for (e in embed_methods) {
print(e)
embedded_data[[e]] <- embed(scurve, e)
for (q in quality_methods)
print(q)
try(quality_results[e, q] <- quality(embedded_data[[e]], q))
}
