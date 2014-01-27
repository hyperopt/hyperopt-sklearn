
from hyperopt.pyll import scope
from .components import (
    pca,
    standard_scaler,
    min_max_scaler,
    normalizer,
    colkmeans,
    rbm,
    )
from hyperopt import hp



def simple_small_image_preprocessing(name):
    """
    Preprocessing appropriate for rasterized small images
    """
    return hp.choice('%s' % name, [
        [pca(name + '.pca')],
        [standard_scaler(name + '.standard_scaler')],
        [min_max_scaler(name + '.min_max_scaler')],
        [normalizer(name + '.normalizer')],
        #[colkmeans(name + '.colkmeans')],
        #[ # -- MinMax -> RBM
        #    min_max_scaler(name + '.mmrbm:min_max_scaler'),
        #    rbm(name + '.mmrbm:rbm')
        #],
        #[ # -- VQ -> RBM
        #    colkmeans(name + '.vqrbm:vq',
        #              n_clusters=scope.int(
        #                  hp.quniform(name + 'vqrbm:vq.n_clusters',
        #                              low=1.51, high=7.49, q=1),
        #                  ),
        #              n_init=1,
        #             ),
        #    rbm(name + '.vqrbm:rbm')
        #],
        #[ # -- VQ -> PCA
        #    colkmeans(name + '.vqpca:vq',
        #              n_clusters=scope.int(
        #                  hp.quniform(name + 'vqpca:vq.n_clusters',
        #                              low=1.51, high=7.49, q=1),
        #                  ),
        #              n_init=1,
        #             ),
        #    pca(name + '.vqpca:pca')
        #],
    ])

