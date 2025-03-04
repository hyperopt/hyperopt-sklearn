from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import feature_extraction
import typing


@scope.define
def sklearn_TfidfVectorizer(*args, **kwargs):
    return feature_extraction.text.TfidfVectorizer(*args, **kwargs)


@scope.define
def sklearn_HashingVectorizer(*args, **kwargs):
    return feature_extraction.text.HashingVectorizer(*args, **kwargs)


@scope.define
def sklearn_CountVectorizer(*args, **kwargs):
    return feature_extraction.text.CountVectorizer(*args, **kwargs)


def _text_analyzer(name: str):
    """
    Declaration search space 'analyzer' parameter
    """
    return hp.choice(name, ["word", "char", "char_wb"])


def _text_stop_words(name: str):
    """
    Declaration search space 'stop_words' parameter
    """
    return hp.choice(name, ["english", None])


def _text_lowercase(name: str):
    """
    Declaration search space 'lowercase' parameter
    """
    return hp.choice(name, [True, False])


def _text_max_df(name: str):
    """
    Declaration search space 'max_df' parameter
    """
    return hp.uniform(name, 0.7, 1.0)


def _text_min_df(name: str):
    """
    Declaration search space 'min_df' parameter
    """
    return hp.uniform(name, 0.0, 0.2)


def _text_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.4, scope.int(hp.uniform(name + ".int", 1, 5))),
        (0.6, None)
    ])


def _text_binary(name: str):
    """
    Declaration search space 'binary' parameter
    """
    return hp.choice(name, [True, False])


def _text_max_ngram(name: str):
    """
    Declaration maximum range for 'ngram_range' parameter
    """
    return scope.int(hp.quniform(name, 1, 4, 1))


def _text_norm(name: str):
    """
    Declaration search space 'norm' parameter
    """
    return hp.choice(name, ["l1", "l2"])


@validate(params=["analyzer"],
          validation_test=lambda param: not isinstance(param, str) or param in ["word", "char", "char_wb"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'word', 'char', 'char_wb'.")
def _text_hp_space(
        name_func,
        analyzer: typing.Union[str, callable, Apply] = None,
        stop_words: typing.Union[str, list, Apply] = None,
        lowercase: typing.Union[bool, Apply] = None,
        binary: typing.Union[bool, Apply] = None,
        ngram_range: tuple = None,
        **kwargs,
):
    """
    Hyper parameter search space for
     tfidf vectorizer
     hashing vectorizer
     count vectorizer
    """
    hp_space = dict(
        analyzer=_text_analyzer(name_func("analyzer")) if analyzer is None else analyzer,
        stop_words=_text_stop_words(name_func("stop_words")) if stop_words is None else stop_words,
        lowercase=_text_lowercase(name_func("lowercase")) if lowercase is None else lowercase,
        binary=_text_binary(name_func("binary")) if binary is None else binary,
        ngram_range=(1, _text_max_ngram(name_func("ngram_range"))) if ngram_range is None else ngram_range,
        **kwargs,
    )
    return hp_space


@validate(params=["norm"],
          validation_test=lambda param: not isinstance(param, str) or param in ["l1", "l2"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'l1', 'l2'.")
def tfidf_vectorizer(
        name: str,
        max_df: typing.Union[float, int, Apply] = 1.0,
        min_df: typing.Union[float, int, Apply] = 1,
        max_features: typing.Union[int, Apply] = None,
        norm: typing.Union[str, Apply] = None,
        **kwargs,
):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.feature_extraction.text.TfidfVectorizer transformer.

    Args:
        name: name | str
        max_df: upper bound document frequency | float
        min_df: lower bound document frequency | float
        max_features: maximum features to consider | int
        norm: 'l1', 'l2' or None | str
    """
    def _name(msg):
        return f"{name}.tfidf_vectorizer_{msg}"

    hp_space = _text_hp_space(_name, **kwargs)
    hp_space["max_df"] = _text_max_df(_name("max_df")) if max_df is None else max_df
    hp_space["min_df"] = _text_min_df(_name("min_df")) if min_df is None else min_df
    hp_space["norm"] = _text_norm(_name("norm")) if norm is None else norm
    hp_space["max_features"] = _text_max_features(_name("max_features")) \
        if max_features is not None else max_features

    return scope.sklearn_TfidfVectorizer(**hp_space)


@validate(params=["norm"],
          validation_test=lambda param: not isinstance(param, str) or param in ["l1", "l2"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'l1', 'l2'.")
def hashing_vectorizer(name: str, norm: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.feature_extraction.text.HashingVectorizer transformer.

    Args:
        name: name | str
        norm: 'l1', 'l2' or None | str
    """
    def _name(msg):
        return f"{name}.hashing_vectorizer_{msg}"

    hp_space = _text_hp_space(_name, **kwargs)
    hp_space["norm"] = _text_norm(_name("norm")) if norm is None else norm

    return scope.sklearn_HashingVectorizer(**hp_space)


def count_vectorizer(
        name: str,
        max_df: typing.Union[float, int, Apply] = 1.0,
        min_df: typing.Union[float, int, Apply] = 1,
        max_features: typing.Union[int, Apply] = None,
        **kwargs,
):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.feature_extraction.text.CountVectorizer transformer.

    Args:
        max_df: upper bound document frequency | float
        min_df: lower bound document frequency | float
        max_features: maximum features to consider | int
    """
    def _name(msg):
        return f"{name}.count_vectorizer_{msg}"

    hp_space = _text_hp_space(_name, **kwargs)
    hp_space["max_df"] = _text_max_df(_name("max_df")) if max_df is None else max_df
    hp_space["min_df"] = _text_min_df(_name("min_df")) if min_df is None else min_df
    hp_space["max_features"] = _text_max_features(_name("max_features")) \
        if max_features is not None else max_features

    return scope.sklearn_CountVectorizer(**hp_space)
