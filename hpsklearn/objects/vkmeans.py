import numpy as np
from sklearn.cluster import KMeans


class ColumnKMeans(object):
    def __init__(self,
                 n_clusters,
                 init="k-means++",
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.output_dtype = None

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        rows, cols = X.shape
        self.col_models = []

        for jj in range(cols):
            col_model = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=self.copy_x,
            )
            col_model.fit(X[:, jj:jj + 1])
            self.col_models.append(col_model)

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        rows, cols = X.shape
        if self.output_dtype is None:
            output_dtype = X.dtype  # XXX
        else:
            output_dtype = self.output_dtype
        rval = np.empty((rows, cols, self.n_clusters),
                        dtype=output_dtype)

        for jj in range(cols):
            Xj = X[:, jj:jj + 1]
            dists = self.col_models[jj].transform(Xj)
            feats = np.exp(-(dists ** 2))
            # -- normalize features by row
            rval[:, jj, :] = feats / (feats.sum(axis=1)[:, None])

        assert np.all(np.isfinite(rval))

        return rval.reshape((rows, cols * self.n_clusters))
