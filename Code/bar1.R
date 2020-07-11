embed_methods <- c("Isomap", "PCA", "tSNE", "UMAP", "kPCA", "LLE", "MDS")
quality_methods <- c("Q_local", "Q_global", "AUC_lnK_R_NX", "cophenetic_correlation")

scurve <- loadDataSet("3D S Curve", n = 2000)
quality_results <- matrix(
NA, length(embed_methods), length(quality_methods),
dimnames = list(embed_methods, quality_methods)
)
embedded_data <- list()
for (e in embed_methods) {
embedded_data[[e]] <- embed(scurve, e)
for (q in quality_methods)
try(quality_results[e, q] <- quality(embedded_data[[e]], q))
}


cs = c(3,4,5,8)
barplot(t(quality_results), beside = TRUE,
        , col = cs, xlim = c(0., 1.1), horiz=TRUE, las=1, width=0.1, border = NA)
legend("topright", x=0.63,y=3.3,
       legend = colnames(quality_results), 
       fill = cs, ncol = 1, cex = 1., bg = "gray90")

