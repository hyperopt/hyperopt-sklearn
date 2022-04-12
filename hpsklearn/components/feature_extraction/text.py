from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import feature_extraction
import typing


@scope.define
def sklearn_TfidfVectorizer(*args, **kwargs):
    return feature_extraction.text.TfidfVectorizer(*args, **kwargs)


@validate(params=["analyzer"],
          validation_test=lambda param: not isinstance(param, str) or param in ["word", "char", "char_wb"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'word', 'char', 'char_wb'.")
@validate(params=["norm"],
          validation_test=lambda param: not isinstance(param, str) or param in ["l1", "l2"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'l1', 'l2'.")
def tfidf(name: str,
          analyzer: typing.Union[str, callable, Apply] = None,
          ngram_range: tuple = None,
          stop_words: typing.Union[str, list, Apply] = None,
          lowercase: typing.Union[bool, Apply] = None,
          max_df: typing.Union[float, int, Apply] = 1.0,
          min_df: typing.Union[float, int, Apply] = 1,
          max_features: int = None,
          binary: typing.Union[bool, Apply] = None,
          norm: typing.Union[str, Apply] = None,
          use_idf: bool = False,
          smooth_idf: bool = False,
          sublinear_tf: bool = False):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.feature_extraction.text.TfidfVectorizer transformer.

    Args:
        name: name | str
        analyzer: features made of word or char n-grams | str, callable
        ngram_range: lower and upper boundary of n values | tuple
        stop_words: stop words | str, list
        lowercase: convert all characters to lowercase | bool
        max_df: upper bound document frequency | float
        min_df: lower bound document frequency | float
        max_features: max number of features | int
        binary: set non-zero term counts to 1 | bool
        norm: 'l1', 'l2' or None | str
        use_idf: enable inverse-document-frequency reweighting | bool
        smooth_idf: smooth idf weights by adding one to document frequencies | bool
        sublinear_tf: apply sublinear tf scaling | bool
    """
    def _name(msg):
        return f"{name}.tfidf_vectorizer_{msg}"

    max_ngram = scope.int(hp.quniform(_name("max_ngram"), 1, 4, 1))

    rval = scope.sklearn_TfidfVectorizer(
        analyzer=hp.choice(_name("analyzer"), ["word", "char", "char_wb"]) if analyzer is None else analyzer,
        stop_words=hp.choice(_name("stop_words"), ["english", None]) if stop_words is None else stop_words,
        lowercase=hp.choice(_name("lowercase"), [True, False]) if lowercase is None else lowercase,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        binary=hp.choice(_name("binary"), [True, False]) if binary is None else binary,
        ngram_range=(1, max_ngram) if ngram_range is None else ngram_range,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )
    return rval
