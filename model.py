"""
Dimensionality Reduction: Compress MNIST with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist_subset
import os
import tempfile
import urllib.request
import numpy as np

def load_mnist_subset(n=3000):
    # Cache mnist.npz in the system temporary directory.
    cache_path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(url, cache_path)

    # Load the first n training images and labels.
    with np.load(cache_path) as data:
        X = data["x_train"][:n]
        y = data["y_train"][:n]

    # Flatten images to (n, 784) and enforce required dtypes.
    X = X.reshape(n, 784).astype(np.float64)
    y = y.astype(np.int64)

    return X, y

# Step 2 - fit_pca
from sklearn.decomposition import PCA

def fit_pca(X, n_components=None, random_state=42):
    # Fit PCA while retaining all components when n_components is None.
    pca = PCA(
        n_components=n_components,
        random_state=random_state
    )
    pca.fit(X)

    return pca

# Step 3 - components_for_variance
def components_for_variance(pca, threshold=0.95):
    # Find the first component where cumulative explained variance
    # reaches or exceeds the requested threshold.
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumulative_variance >= threshold)) + 1

# Step 4 - explained_variance_curve
def explained_variance_curve(pca, ks):
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return {
        k: round(float(cumulative_variance[k - 1]), 4)
        for k in ks
    }

# Step 5 - compress
def compress(X, n_components, random_state=42):
    pca = fit_pca(
        X,
        n_components=n_components,
        random_state=random_state
    )
    X_reduced = pca.transform(X)

    return pca, X_reduced

# Step 6 - reconstruction_error
def reconstruction_error(pca, X):
    X_rec = pca.inverse_transform(pca.transform(X))
    return float(np.mean((X - X_rec) ** 2))

# Step 7 - compression_ratio
def compression_ratio(X, X_reduced):
    return round(float(X.shape[1] / X_reduced.shape[1]), 2)

# Step 8 - randomized_pca
def randomized_pca(X, n_components, random_state=42):
    # Fit PCA using the randomized SVD solver.
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state
    )
    pca.fit(X)

    # Compare its explained variance with the exact PCA.
    exact_pca, _ = compress(X, n_components, random_state)

    variance_gap = float(
        abs(
            np.sum(pca.explained_variance_ratio_)
            - np.sum(exact_pca.explained_variance_ratio_)
        )
    )

    return {
        "pca": pca,
        "variance_gap": variance_gap
    }

# Step 9 - incremental_pca
from sklearn.decomposition import IncrementalPCA

def incremental_pca(X, n_components, n_batches=10):
    ipca = IncrementalPCA(n_components=n_components)

    for batch in np.array_split(X, n_batches):
        ipca.partial_fit(batch)

    return ipca

# Step 10 - random_projection
from sklearn.random_projection import GaussianRandomProjection

def random_projection(X, n_components, random_state=42, n_pairs=500):
    # Fit Gaussian random projection and transform the data.
    projector = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state
    )
    X_reduced = projector.fit_transform(X)

    # Generate random index pairs and skip pairs with identical indices.
    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), (n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    # Compute projected/original Euclidean distance ratios.
    original_distances = np.linalg.norm(
        X[pairs[:, 0]] - X[pairs[:, 1]],
        axis=1
    )
    projected_distances = np.linalg.norm(
        X_reduced[pairs[:, 0]] - X_reduced[pairs[:, 1]],
        axis=1
    )

    ratios = projected_distances / original_distances

    return {
        "X_reduced": X_reduced,
        "mean_ratio": float(np.mean(ratios)),
        "max_distortion": float(np.max(np.abs(ratios - 1)))
    }

# Step 11 - unroll_swiss_roll
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

def unroll_swiss_roll(n_samples=1000, random_state=42):
    # Generate the 3-D Swiss roll and its position parameter t.
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.2,
        random_state=random_state
    )

    # Flatten the Swiss roll into a 2-D embedding using LLE.
    lle = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
        random_state=random_state
    )
    X_2d = lle.fit_transform(X)

    # Compute the absolute Pearson correlation between t and each
    # embedding coordinate, then keep the larger correlation.
    correlations = [
        abs(np.corrcoef(t, X_2d[:, i])[0, 1])
        for i in range(X_2d.shape[1])
    ]

    t_correlation = float(max(correlations))

    return {
        "X_2d": X_2d,
        "t_correlation": t_correlation
    }

# Step 12 - tsne_map
from sklearn.manifold import TSNE

def tsne_map(X, y, n=1000, random_state=42):
    # Use the first n samples.
    X_subset = X[:n]
    y_subset = y[:n]

    # Embed the digits into 2-D using t-SNE.
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=random_state
    )
    X_2d = tsne.fit_transform(X_subset)

    # Compute class centroids.
    classes = np.unique(y_subset)
    centroids = np.array([
        X_2d[y_subset == cls].mean(axis=0)
        for cls in classes
    ])

    # Mean distance between all pairs of distinct class centroids.
    centroid_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            centroid_distances.append(
                np.linalg.norm(centroids[i] - centroids[j])
            )

    mean_between = float(np.mean(centroid_distances))

    # Mean within-class distance from samples to their class centroid.
    within_distances = []
    for i, cls in enumerate(classes):
        class_points = X_2d[y_subset == cls]
        distances = np.linalg.norm(class_points - centroids[i], axis=1)
        within_distances.extend(distances)

    mean_within = float(np.mean(within_distances))

    separation = float(mean_between / mean_within)

    return {
        "X_2d": X_2d,
        "separation": separation
    }

# Step 13 - classifier_on_compressed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier_on_compressed(
    X, y, n_components, test_size=0.25, random_state=42
):
    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Train logistic regression on the raw pixel features.
    raw_clf = LogisticRegression(max_iter=2000)
    raw_clf.fit(X_train, y_train)
    raw_accuracy = float(raw_clf.score(X_test, y_test))

    # Fit PCA only on the training data to avoid data leakage.
    pca = fit_pca(
        X_train,
        n_components=n_components,
        random_state=random_state
    )

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train the same classifier on the PCA-compressed features.
    pca_clf = LogisticRegression(max_iter=2000)
    pca_clf.fit(X_train_pca, y_train)
    pca_accuracy = float(pca_clf.score(X_test_pca, y_test))

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (X.shape[1], X_train_pca.shape[1])
    }

# Step 14 - pca_classifier_pipeline
from sklearn.pipeline import make_pipeline

def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(
            n_components=variance,
            random_state=random_state
        ),
        LogisticRegression(max_iter=2000)
    )

# Step 15 - save_and_reload_pipeline
import joblib

def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)

# Step 16 - predict_images (not yet solved)
# TODO: implement

