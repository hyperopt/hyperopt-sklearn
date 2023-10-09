from ._pfit import _pfit_until_convergence, retrain_fraction
from ._transform import _transform_combine_XEX
from ._cost_fn import _cost_fn

from hyperopt import pyll
import hyperopt.rand

from multiprocessing import Process, Pipe
from sklearn.base import BaseEstimator
from functools import partial

import scipy.sparse
import numpy as np
import numpy.typing as npt
import pathlib
import inspect
import typing
import pickle


class hyperopt_estimator(BaseEstimator):
    """
    Hyperopt estimator class
     Used for fitting and scoring
    """
    def __init__(self,
                 preprocessing: typing.Union[pyll.Apply, list] = None,
                 ex_preprocs: typing.Union[list, tuple] = None,
                 classifier: pyll.Apply = None,
                 regressor: pyll.Apply = None,
                 space: typing.Union[pyll.Apply, dict] = None,
                 algo: callable = hyperopt.rand.suggest,
                 max_evals: int = 10,
                 loss_fn: callable = None,
                 continuous_loss_fn: bool = False,
                 verbose=False,
                 trial_timeout: typing.Optional[float] = None,
                 fit_increment: int = 1,
                 fit_increment_dump_filename: typing.Union[str, pathlib.Path] = None,
                 seed: typing.Union[np.random.Generator, int] = None,
                 use_partial_fit: bool = False,
                 refit: bool = True,
                 n_jobs: int = 1
                 ) -> None:
        """
        Parameters
        ----------

        preprocessing: pyll.Apply node, default is None
            This should evaluate to a list of sklearn-style preprocessing
            modules (may include hyperparameters). When None, a random
            preprocessing module will be used.

        ex_preprocs: pyll.Apply node, default is None
            This should evaluate to a list of lists of sklearn-style
            preprocessing modules for each exogenous dataset. When None, no
            preprocessing will be applied to exogenous data.

        classifier: pyll.Apply node
            This should evaluate to sklearn-style classifier. (may include
            hyperparameters)

        regressor: pyll.Apply node
            This should evaluate to sklearn-style regressor. (may include
            hyperparameters)

        space: pyll.Apply node
            Hyperopt search space to be used in the optimization process.

        algo: callable, default is hyperopt.rand.suggest
            Hyperopt suggest algo. (e.g. rand.suggest)

        max_evals: int, default is 10
            Fit() will evaluate up to this-many configurations. Does not apply
            to fit_iter, which continues to search indefinitely.

        loss_fn: callable, defaults to hpsklearn _cost_fn
            A function that takes the arguments (y_target, y_prediction)
            and computes a loss value to be minimized. If no function is
            specified, '1.0 - accuracy_score(y_target, y_prediction)' is used
            for classification and '1.0 - r2_score(y_target, y_prediction)'
            is used for regression.

        continuous_loss_fn: boolean, default is False
            When true, the loss function is passed the output of
            predict_proba() as the second argument.  This is to facilitate the
            use of continuous loss functions like cross entropy or AUC.  When
            false, the loss function is given the output of predict().  If
            true, `classifier` and `loss_fn` must also be specified.

        verbose: boolean, default is False
            When true, info will be printed during the process.

        trial_timeout: float (seconds), or None for no timeout, default is None
            Kill trial evaluations after this many seconds.

        fit_increment: int, default is 1
            Every this-many trials will be a synchronization barrier for
            ongoing trials, and the hyperopt Trials object may be
            check-pointed.  (Currently evaluations are done serially, but
            that might easily change in future to allow e.g. MongoTrials)

        fit_increment_dump_filename : str or None, default is None
            Periodically dump self.trials to this file (via cPickle) during
            fit(). Saves after every `fit_increment` trial evaluations.

        seed: numpy.random.Generator or int or None, default is None
            If int, the integer will be used to seed a Generator instance
            for use in hyperopt.fmin. Use None to make sure each run is
            independent. Default is None.

        use_partial_fit : boolean, default is False
            If the learner support partial fit, it can be used for online
            learning. However, the whole train set is not split into mini
            batches here. The partial fit is used to iteratively update
            parameters on the whole train set. Early stopping is used to kill
            the training when the validation score stops improving.

        refit: boolean, default is True
            Refit the best model on the whole data set.

        n_jobs: integer, default is 1
            Use multiple CPU cores when training estimators which support
            multiprocessing.
        """
        self.preprocessing = preprocessing
        self.ex_preprocs = ex_preprocs
        self.classifier = classifier
        self.regressor = regressor
        self.space = space
        self.algo = algo
        self.max_evals = max_evals
        self.loss_fn = loss_fn
        self.continuous_loss_fn = continuous_loss_fn
        self.verbose = verbose
        self.trial_timeout = trial_timeout
        self.fit_increment = fit_increment
        self.fit_increment_dump_filename = fit_increment_dump_filename
        self.seed = seed
        self.use_partial_fit = use_partial_fit
        self.refit = refit
        self.n_jobs = n_jobs
        self._times_fitted = 0

    def _init(self):
        self._best_preprocs = ()
        self._best_ex_preprocs = ()
        self._best_learner = None
        self._best_loss = self._best_loss = float('inf')
        self._best_iters = None
        self._times_fitted += 1

        if self.space is None:
            assert not all(isinstance(v, pyll.Apply) for v in [self.regressor, self.classifier])

            if self.classifier is None and self.regressor is None:
                from hpsklearn import any_classifier
                self.classifier = any_classifier(name="classifier")

            self.classification = isinstance(self.classifier, pyll.Apply)
            self.classification = True if self.classifier is not None else False

            if self.preprocessing is None:
                from hpsklearn import any_preprocessing
                self.preprocessing = any_preprocessing(name="preprocessing")

            if self.ex_preprocs is None:
                self.ex_preprocs = list()

            self.space = pyll.as_apply({
                "classifier": self.classifier,
                "regressor": self.regressor,
                "preprocessing": self.preprocessing,
                "ex_preprocs": self.ex_preprocs
            })
        else:
            if self._times_fitted == 1:
                assert all(v is None for v in [self.classifier, self.regressor, self.preprocessing,
                                               self.ex_preprocs]), \
                    "Detected a search space. " \
                    "Parameters 'classifier', 'regressor', 'preprocessing' and 'ex_preprocs' " \
                    "should be contained in the space and should not be set in addition to the space."

            if isinstance(self.space, dict):
                self.space = pyll.as_apply(self.space)

            eval_space = dict(self.space.named_args)
            assert all(k in eval_space.keys() for k in ["classifier", "regressor", "preprocessing"]), \
                "Detected a search space. " \
                "Parameters 'classifier', 'regressor' and 'preprocessing' should be supplied. " \
                "'None' suffices for empty parameters."

            if "ex_preprocs" in eval_space.keys():
                self.ex_preprocs = eval_space["ex_preprocs"].eval()
            else:
                self.ex_preprocs = list()
                self.space.named_args.append(["ex_preprocs", pyll.as_apply(self.ex_preprocs)])

        self.n_ex_pps = len(self.ex_preprocs)
        self.algo = self.algo or hyperopt.rand.suggest
        self.rstate = np.random.default_rng(self.seed)

        if self.continuous_loss_fn:
            assert self.space['classifier'] is not None, "Can only use 'continuous_loss_fn' with classifiers."
            assert self.loss_fn is not None, "Must specify 'loss_fn' if 'continuous_loss_fn' is true."

    def info(self, *args) -> None:
        """
        If verbose, print info during optimization process
        """
        if self.verbose:
            print(" ".join(map(str, args)))

    def fit_iter(self, X, y,
                 EX_list: typing.Union[list, tuple] = None,
                 valid_size: float = .2,
                 n_folds: int = None,
                 kfolds_group: typing.Union[list, np.ndarray] = None,
                 cv_shuffle: bool = False,
                 warm_start: bool = False,
                 random_state: np.random.Generator = np.random.default_rng(),
                 ) -> typing.Generator:
        """
        Generator of Trials after ever-increasing numbers of evaluations

        Args:
            X:
                Input variables

            y:
                Output variables

            EX_list: list, default is None
                List of exogenous datasets. Each must have the same number of
                samples as X.

            valid_size: float, default is 0.2
                The portion of the dataset used as the validation set. If
                cv_shuffle is False, always use the last samples as validation.

            n_folds: int, default is None
                When n_folds is not None, use K-fold cross-validation when
                n_folds > 2. Or, use leave-one-out cross-validation when
                n_folds = -1. For Group K-fold cross-validation, functions as
                `n_splits`.

            kfolds_group: list or ndarray, default is None
                When kfolds_group is not None, use Group K-fold cross-validation
                with the specified groups. The length of kfolds_group must be
                equal to the number of samples in X.

            cv_shuffle: bool, default is False
                Whether to perform sample shuffling before splitting the
                data into training and validation sets.

            warm_start: bool, default is False
                If True, the estimator will start from an existing sequence
                of trials.

            random_state: Generator, default is np.random.default_rng()
                The random state used to seed the cross-validation shuffling.
        """
        if self._times_fitted == 0:
            self._init()

        increment = self.fit_increment

        # Convert list, pandas series, or other array-like to ndarray
        # do not convert sparse matrices
        if not scipy.sparse.issparse(X) and not isinstance(X, np.ndarray):
            X = np.array(X)

        if not scipy.sparse.issparse(y) and not isinstance(y, np.ndarray):
            y = np.array(y)

        if not warm_start:
            self.trials = hyperopt.Trials()
            self._best_loss = float("inf")
        else:
            assert hasattr(self, "trials"), "If 'warm_start', a 'trials' parameter must be supplied."

        fn = partial(_cost_fn,
                     X=X, y=y,
                     EX_list=EX_list,
                     valid_size=valid_size,
                     n_folds=n_folds,
                     kfolds_group=kfolds_group,
                     shuffle=cv_shuffle,
                     random_state=random_state,
                     use_partial_fit=self.use_partial_fit,
                     info=self.info,
                     timeout=self.trial_timeout,
                     loss_fn=self.loss_fn,
                     continuous_loss_fn=self.continuous_loss_fn,
                     n_jobs=self.n_jobs
                     )

        # Wrap up the cost function as a process with timeout control.
        def _fn_with_timeout(*args, **kwargs):
            conn1, conn2 = Pipe()
            kwargs["_conn"] = conn2
            th = Process(target=partial(fn, best_loss=self._best_loss),
                         args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(self.trial_timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                self.info("TERMINATING DUE TO TIME-OUT.")
                th.terminate()
                th.join()
                fn_rval = "return", {
                    "status": hyperopt.STATUS_FAIL,
                    "failure": "TimeOut"
                }

            assert fn_rval[0] in ("raise", "return")
            if fn_rval[0] == "raise":
                raise fn_rval[1]

            # -- remove potentially large objects from the rval
            #    so that the Trials() object below stays small
            #    We can recompute them if necessary, and it's usually
            #    not necessary at all.
            if fn_rval[1]["status"] == hyperopt.STATUS_OK:
                fn_loss = float(fn_rval[1].get("loss"))
                fn_preprocs = fn_rval[1].pop("preprocs")
                fn_ex_preprocs = fn_rval[1].pop("ex_preprocs")
                fn_learner = fn_rval[1].pop("learner")
                fn_iters = fn_rval[1].pop("iterations")
                if fn_loss < self._best_loss:
                    self._best_preprocs = fn_preprocs
                    self._best_ex_preprocs = fn_ex_preprocs
                    self._best_learner = fn_learner
                    self._best_loss = fn_loss
                    self._best_iters = fn_iters
            return fn_rval[1]

        while True:
            new_increment = yield self.trials

            if new_increment is not None:
                increment = new_increment

            # Workaround for rstate issue #35
            if "rstate" in inspect.getfullargspec(hyperopt.fmin).args:
                hyperopt.fmin(_fn_with_timeout,
                              space=self.space,
                              algo=self.algo,
                              trials=self.trials,
                              max_evals=len(self.trials.trials) + increment,
                              # -- let exceptions crash the program, so we notice them.
                              catch_eval_exceptions=False,
                              return_argmin=False)  # -- in case no success so far
            else:
                if self.seed is None:
                    hyperopt.fmin(_fn_with_timeout,
                                  space=self.space,
                                  algo=self.algo,
                                  trials=self.trials,
                                  max_evals=len(self.trials.trials) + increment)
                else:
                    hyperopt.fmin(_fn_with_timeout,
                                  space=self.space,
                                  algo=self.algo,
                                  trials=self.trials,
                                  max_evals=len(self.trials.trials) + increment,
                                  rstate=self.rstate)

    def _retrain_best_model_on_full_data(self, X, y,
                                         EX_list: typing.Union[list, tuple] = None) -> None:
        """
        Retrain best model located in self._best_learner
         on the full dataset.

        Args:
            X:
                Input variables

            y:
                Output variables

            EX_list: list, default is None
                List of exogenous datasets. Each must have the same number of
                samples as X.
        """
        if EX_list is not None:
            assert len(EX_list) == self.n_ex_pps

        XEX = _transform_combine_XEX(
            X, info=self.info, en_pps=self._best_preprocs,
            EXfit_list=EX_list, ex_pps_list=self._best_ex_preprocs
        )

        self.info(f"Training learner {self._best_learner} on X/EX of dimension {XEX.shape}")
        if hasattr(self._best_learner, "partial_fit") and self.use_partial_fit:
            self._best_learner, _ = _pfit_until_convergence(
                learner=self._best_learner, is_classif=self.classification,
                XEXfit=XEX, yfit=y, info=self.info,
                max_iters=int(self._best_iters * retrain_fraction)
            )
        else:
            self._best_learner.fit(XEX, y)

    def fit(self, X, y,
            EX_list: typing.Union[list, tuple] = None,
            valid_size: float = .2,
            n_folds: int = None,
            kfolds_group: typing.Union[list, np.ndarray] = None,
            cv_shuffle: bool = False,
            warm_start: bool = False,
            random_state: np.random.Generator = np.random.default_rng()
            ) -> None:
        """
        Search the space of learners and preprocessing steps for a good
        predictive model of y <- X. Store the best model for predictions.

        Args:
            X:
                Input variables

            y:
                Output variables

            EX_list: list, default is None
                List of exogenous datasets. Each must have the same number of
                samples as X.

            valid_size: float, default is 0.2
                The portion of the dataset used as the validation set. If
                cv_shuffle is False, always use the last samples as validation.

            n_folds: int, default is None
                When n_folds is not None, use K-fold cross-validation when
                n_folds > 2. Or, use leave-one-out cross-validation when
                n_folds = -1. For Group K-fold cross-validation, functions as
                `n_splits`.

            kfolds_group: list or ndarray, default is None
                When kfolds_group is not None, use Group K-fold cross-validation
                with the specified groups. The length of group_kfolds must be
                equal to the number of samples in X.

            cv_shuffle: bool, default is False
                Whether to perform sample shuffling before splitting the
                data into training and validation sets.

            warm_start: bool, default is False
                If True, the estimator will start from an existing sequence
                of trials.

            random_state: Generator, default is np.random.default_rng()
                The random state used to seed the cross-validation shuffling.

        Notes:
            For classification problems, hpsklearn will always use the stratified
            version of the K-fold cross-validation or shuffle-and-split.
        """
        self._init()

        if EX_list is not None:
            assert len(EX_list) == self.n_ex_pps

        fit_iter = self.fit_iter(X=X, y=y,
                                 EX_list=EX_list,
                                 valid_size=valid_size,
                                 n_folds=n_folds,
                                 kfolds_group=kfolds_group,
                                 cv_shuffle=cv_shuffle,
                                 warm_start=warm_start,
                                 random_state=random_state)

        next(fit_iter)
        adjusted_max_evals = self.max_evals if not warm_start else len(self.trials.trials) + self.max_evals

        while len(self.trials.trials) < adjusted_max_evals:
            try:
                increment = min(self.fit_increment,
                                adjusted_max_evals - len(self.trials.trials))
                fit_iter.send(increment)

                if self.fit_increment_dump_filename is not None:
                    with open(self.fit_increment_dump_filename, "wb") as dump_file:
                        self.info("---> dumping trials to", self.fit_increment_dump_filename)
                        pickle.dump(self.trials, dump_file)
            except KeyboardInterrupt:
                break

        if self._best_learner is None:
            raise RuntimeError(
                "All trials failed or timed out. \n"
                f"Result of last trial: {self.trials.trials[-1]['result']}"
            )

        if self.refit:
            self._retrain_best_model_on_full_data(X, y, EX_list=EX_list)

    def predict(self, X,
                EX_list: typing.Union[list, tuple] = None,
                fit_preproc: bool = False) -> npt.ArrayLike:
        """
        Use the best model found by previous fit() to make a prediction.
        """
        if self._best_learner is None:
            raise RuntimeError(
                "Attempting to use a model that has not been fit. "
                "Ensure fit() has been called and at least one trial "
                "has completed without failing or timing out."
            )

        if EX_list is not None:
            assert len(EX_list) == self.n_ex_pps

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X)
        else:
            X = np.array(X)
        XEX = _transform_combine_XEX(
            X, info=self.info, en_pps=self._best_preprocs,
            EXfit_list=EX_list, ex_pps_list=self._best_ex_preprocs,
            fit_preproc=fit_preproc,
        )
        return self._best_learner.predict(XEX)

    def score(self, X, y,
              EX_list: typing.Union[list, tuple] = None,
              fit_preproc: bool = False) -> float:
        """
        Return the score (accuracy or R2) of the learner on
        a given set of data

        Args:
            X:
                Input variables

            y:
                Output variables

            EX_list: list, default is None
                List of exogenous datasets. Each must have the same number of
                samples as X.

            fit_preproc: bool, default is False
                Whether to fit the preprocessing algorithm
        """
        if self._best_learner is None:
            raise RuntimeError(
                "Attempting to use a model that has not been fit. "
                "Ensure fit() has been called and at least one trial "
                "has completed without failing or timing out."
            )

        if EX_list is not None:
            assert len(EX_list) == self.n_ex_pps

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X)
        else:
            X = np.array(X)
        XEX = _transform_combine_XEX(
            X, info=self.info, en_pps=self._best_preprocs,
            EXfit_list=EX_list, ex_pps_list=self._best_ex_preprocs,
            fit_preproc=fit_preproc,
        )

        return self._best_learner.score(XEX, y)

    def best_model(self) -> dict:
        """
        Returns the best model found by the previous fit()
        """
        return {"learner": self._best_learner,
                "preprocs": self._best_preprocs,
                "ex_preprocs": self._best_ex_preprocs}
