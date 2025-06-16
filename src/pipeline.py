import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline for imputing, scaling, and selecting clinical features using ElasticNet.
    """
    def __init__(self):
        self.pipeline = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=-1),
            StandardScaler(),
            ElasticNetCV(cv=5, random_state=42)
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        coefs = self.pipeline.named_steps['elasticnetcv'].coef_
        self.selected_columns_ = X.columns[np.abs(coefs) > 0]
        return self

    def transform(self, X):
        return self.pipeline.transform(X[self.selected_columns_])


class WordFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates TF-IDF features filtered by Mutual Information.
    """
    def __init__(self, mi_threshold=0.005):
        self.mi_threshold = mi_threshold
        self.vectorizer = TfidfVectorizer(use_idf=True, min_df=5, sublinear_tf=True)

    def fit(self, X, y):
        from sklearn.feature_selection import mutual_info_classif
        X_vec = self.vectorizer.fit_transform(X)
        mi_scores = mutual_info_classif(X_vec, y, discrete_features=False, random_state=42)
        self.selected_indices_ = np.where(mi_scores > self.mi_threshold)[0]
        return self

    def transform(self, X):
        X_vec = self.vectorizer.transform(X)
        return X_vec[:, self.selected_indices_].toarray()


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes input text using a pretrained SentenceTransformer model.
    """
    def __init__(self, model_name='intfloat/multilingual-e5-base', device='cuda'):
        self.model_name = model_name
        self.device = device

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        self.model_.to(self.device)
        return self

    def transform(self, X):
        texts = X.tolist()
        embeddings = []
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc='Encoding embeddings'):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model_.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        return np.vstack(embeddings)


def build_pipeline(clinical_columns, text_column, embedding_column):
    """
    Builds unified feature pipeline combining clinical, text, and embedding features.
    """
    return FeatureUnion([
        ('clinical', ClinicalPreprocessor()),
        ('words', WordFeatureGenerator()),
        ('embeddings', EmbeddingTransformer())
    ])


class LGBMModel:
    """
    Wrapper for LightGBM model with balanced class weights.
    """
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            class_weight={0: 1, 1: 2},
            importance_type='gain',
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model