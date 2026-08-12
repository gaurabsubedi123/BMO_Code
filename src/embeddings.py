"""
embeddings.py -- Shared sklearn-compatible transformer classes for pretrained embeddings.

Imported by 02a, 02b, and demo/app.py so that pickled pipelines can be
unpickled in any of those contexts without AttributeError.

Both transformers access their models via module-level globals (FT_MODEL /
BWV_MODEL) rather than storing them as constructor parameters. This is
necessary because fasttext_pybind and gensim KeyedVectors objects are not
deepcopy-safe (fasttext) or would bloat pickles (gensim), and sklearn's
GridSearchCV calls deepcopy on the full pipeline before each fit.

The globals are set by the calling script (02a / 02b) or by demo/app.py
before the transformers are used.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# Module-level model references — set by the importing script before use.
# Transformers read these at transform() time, never at __init__ time.
FT_MODEL  = None   # gensim KeyedVectors — set by 02a or demo/app.py
BWV_MODEL = None   # fasttext model      — set by 02b or demo/app.py


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """Average pretrained FastText word vectors across all tokens in a document.

    Accesses FT_MODEL from this module's globals — not stored as a constructor
    parameter — so sklearn can safely deepcopy the transformer during CV.
    """

    def __init__(self, dim=300):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model   = FT_MODEL
        vectors = []
        for text in X:
            words     = str(text).lower().split()
            word_vecs = []
            if model is not None:
                for w in words:
                    try:
                        word_vecs.append(model[w])
                    except KeyError:
                        pass
            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0).astype(np.float32))
            else:
                vectors.append(np.zeros(self.dim, dtype=np.float32))
        return csr_matrix(np.vstack(vectors))


class BioWordVecVectorizer(BaseEstimator, TransformerMixin):
    """Average pretrained BioWordVec word vectors across all tokens in a document.

    Accesses BWV_MODEL from this module's globals — not stored as a constructor
    parameter — so sklearn can safely deepcopy the transformer during CV.

    Uses fasttext's get_word_vector() which handles OOV via character subword
    n-grams, so no tokens are ever skipped.
    """

    def __init__(self, dim=200):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model   = BWV_MODEL
        vectors = []
        for text in X:
            words = str(text).lower().split()
            if words and model is not None:
                word_vecs = [model.get_word_vector(w) for w in words]
                vectors.append(np.mean(word_vecs, axis=0).astype(np.float32))
            else:
                vectors.append(np.zeros(self.dim, dtype=np.float32))
        return csr_matrix(np.vstack(vectors))
