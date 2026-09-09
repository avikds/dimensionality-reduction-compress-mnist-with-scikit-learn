# Dimensionality Reduction: Compress MNIST with Scikit-Learn

Chapter 7 of Hands-On Machine Learning as a practitioner applies it: fit PCA on MNIST, choose the number of components from the explained-variance curve, compress and reconstruct images and measure what was lost, speed things up with randomized and incremental PCA, compare with a random projection, unroll a Swiss roll with LLE and map digits in 2-D with t-SNE, show that a classifier on the compressed features keeps its accuracy, then save a PCA-plus-classifier pipeline and serve it on raw images.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_mnist_subset
- [x] **2.** fit_pca
- [x] **3.** components_for_variance
- [x] **4.** explained_variance_curve
- [x] **5.** compress
- [x] **6.** reconstruction_error
- [x] **7.** compression_ratio
- [x] **8.** randomized_pca
- [x] **9.** incremental_pca
- [x] **10.** random_projection
- [x] **11.** unroll_swiss_roll
- [x] **12.** tsne_map
- [x] **13.** classifier_on_compressed
- [x] **14.** pca_classifier_pipeline
- [x] **15.** save_and_reload_pipeline
- [x] **16.** predict_images

## Results

```
MNIST slice (3000, 784); cumulative explained variance: 1:0.099  10:0.500  50:0.832  145:0.951  300:0.989  784:1.000
95% of the variance needs 145 of 784 components
   10 components: 78.40x smaller, reconstruction MSE 2,177.0 per pixel
   50 components: 15.68x smaller, reconstruction MSE 729.7 per pixel
  145 components:  5.41x smaller, reconstruction MSE 216.7 per pixel
randomized solver: explained-variance gap vs exact 0.00000
incremental PCA in 10 batches: reconstruction MSE 219.4
random projection to 300 dims: mean distance ratio 1.015, max distortion 0.127

LLE unrolls the Swiss roll: |corr(t, embedding)| = 0.997
t-SNE map of 1000 digits: between/within class separation 2.54

logistic regression: raw 784 pixels -> 0.889; PCA 145 features -> 0.877
served pipeline (145 components) on 6 raw images: [7, 2, 1, 0, 4, 1] (truth [7, 2, 1, 0, 4, 1])
```
