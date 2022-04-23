from ._utils import _NonFiniteFeature

import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA


def _transform_combine_XEX(Xfit,
                           info: callable = print,
                           en_pps=None,
                           Xval=None,
                           EXfit_list: list = None,
                           ex_pps_list=None,
                           EXval_list: list = None,
                           fit_preproc: bool = True):
    """
    Transform endogenous and exogenous datasets and combine them into a
    single dataset for training and testing.

    Args:
        Xfit:
            Indices of X from endogenous dataset.

        info: callable, default is print
            Callable to handle information with during the loss calculation
            process.

        en_pps: default is None
            This should evaluate to a list of sklearn-style preprocessing
            modules (may include hyperparameters). When None, a random
            preprocessing module will be used.

        Xval: default is None
            Values of X from endogenous dataset.

        EXfit_list: list, default is None
            Indices of X from exogenous dataset.

        ex_pps_list: default is None
            This should evaluate to a list of lists of sklearn-style
            preprocessing modules for each exogenous dataset. When None, no
            preprocessing will be applied to exogenous data.

        EXval_list: list, default is None
            Values of X from exogenous dataset.

        fit_preproc: bool, default is True
            Whether to fit the preprocessing algorithm
    """
    if not ex_pps_list:
        ex_pps_list = list()

    if not en_pps:
        en_pps = list()

    transformed_XEX_list = list()
    en_pps_list, ex_pps_list = list(en_pps), list(ex_pps_list)

    if not ex_pps_list and EXfit_list is not None:
        ex_pps_list = [[]] * len(EXfit_list)
    xex_pps_list = [en_pps_list] + ex_pps_list

    if EXfit_list is None:
        EXfit_list = list()
        assert EXval_list is None
        EXval_list = list()
    elif EXval_list is None:
        EXval_list = [None] * len(EXfit_list)

    EXfit_list, EXval_list = list(EXfit_list), list(EXval_list)
    XEXfit_list, XEXval_list = [Xfit] + EXfit_list, [Xval] + EXval_list

    for pps, dfit, dval in zip(xex_pps_list, XEXfit_list, XEXval_list):
        if pps:
            dfit, dval = _run_preprocs(preprocessing=pps,
                                       Xfit=dfit, Xval=dval,
                                       info=info, fit_preproc=fit_preproc)
        if dval is not None:
            transformed_XEX_list.append((dfit, dval))
        else:
            transformed_XEX_list.append(dfit)

    if Xval is None:
        XEXfit = _safe_concatenate(transformed_XEX_list)
        return XEXfit
    else:
        XEXfit_list, XEXval_list = zip(*transformed_XEX_list)
        XEXfit = _safe_concatenate(XEXfit_list)
        XEXval = _safe_concatenate(XEXval_list)
        return XEXfit, XEXval


def _run_preprocs(preprocessing: list,
                  Xfit,
                  Xval=None,
                  info: callable = print,
                  fit_preproc: bool = True):
    """
    Run all preprocessing steps in a pipeline

    Args:
        preprocessing: list
            All preprocessing steps

        Xfit:
            Indices of X from combined endogenous and exogenous
            datasets.

        Xval:
            Values of X from combined endogenous and exogenous
            datasets.

        info: callable, default is print
            Callable to handle information with during the loss
            calculation process.

        fit_preproc: bool, default is True
            Whether to fit the preprocessing algorithm
    """
    for pp_algo in preprocessing:
        info(f"Fitting {pp_algo} to X of shape {Xfit.shape}")
        if isinstance(pp_algo, PCA):
            n_components = pp_algo.get_params()["n_components"]
            n_components = min([n_components] + list(Xfit.shape))
            pp_algo.set_params(n_components=n_components)
            info(f"Limited PCA n_components at {n_components}")

        if fit_preproc:
            pp_algo.fit(Xfit)

        info(f"Transforming Xfit {Xfit.shape}")
        Xfit = pp_algo.transform(Xfit)

        # np.isfinite() does not work on sparse matrices
        if not (scipy.sparse.issparse(Xfit) or np.all(np.isfinite(Xfit))):
            raise _NonFiniteFeature(pp_algo)

        if Xval is not None:
            info(f"Transforming Xval {Xval.shape}")
            Xval = pp_algo.transform(Xval)
            if not (scipy.sparse.issparse(Xval) or np.all(np.isfinite(Xval))):
                raise _NonFiniteFeature(pp_algo)

    return Xfit, Xval


def _safe_concatenate(XS: list):
    """
    Safely concatenate lists
     use np.concatenate or scipy concatenation

    Args:
        XS: list
            List to concatenate.
    """
    if not any(scipy.sparse.issparse(x) for x in XS):
        return np.concatenate(XS, axis=1)

    XS = [x if scipy.sparse.issparse(x) else scipy.sparse.csr_matrix(x) for x in XS]

    return scipy.sparse.hstack(XS)
