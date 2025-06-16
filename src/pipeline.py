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
    Selects informative clinical features using ElasticNet, then scales them.
    """
    def __init__(self, clinical_columns=None):
        self.clinical_columns = clinical_columns
        self.imputer = SimpleImputer(strategy='constant', fill_value=-1)
        self.scaler = StandardScaler()
        self.selector = ElasticNetCV(cv=5, random_state=42)

    def fit(self, X, y):
        X_clinical = X[self.clinical_columns]
        X_imputed = self.imputer.fit_transform(X_clinical)
        self.selector.fit(X_imputed, y)

        selected_mask = np.abs(self.selector.coef_) > 0
        self.selected_indices_ = np.where(selected_mask)[0]
        self.selected_columns_ = np.array(self.clinical_columns)[self.selected_indices_]

        X_selected = X_imputed[:, self.selected_indices_]
        self.scaler.fit(X_selected)

        return self

    def transform(self, X):
        X_clinical = X[self.clinical_columns]
        X_imputed = self.imputer.transform(X_clinical)
        X_selected = X_imputed[:, self.selected_indices_]
        return self.scaler.transform(X_selected)


class WordFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates TF-IDF features filtered by Mutual Information from a specified text column.
    """
    def __init__(self, text_column, mi_threshold=0.005):
        self.text_column = text_column
        self.mi_threshold = mi_threshold
        self.vectorizer = TfidfVectorizer(use_idf=True, min_df=5, sublinear_tf=True)

    def fit(self, X, y):
        from sklearn.feature_selection import mutual_info_classif
        texts = X[self.text_column].tolist()
        X_vec = self.vectorizer.fit_transform(texts)
        # ✅ Fix: use discrete_features=True for sparse matrix
        mi_scores = mutual_info_classif(X_vec, y, discrete_features=True, random_state=42)
        self.selected_indices_ = np.where(mi_scores > self.mi_threshold)[0]
        return self

    def transform(self, X):
        texts = X[self.text_column].tolist()
        X_vec = self.vectorizer.transform(texts)
        return X_vec[:, self.selected_indices_].toarray()


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes input text using a pretrained SentenceTransformer model.
    """
    def __init__(self, text_column, model_name='intfloat/multilingual-e5-base', device='cpu'):
        self.text_column = text_column
        self.model_name = model_name
        self.device = device

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        self.model_.to(self.device)
        return self

    def transform(self, X):
        texts = X[self.text_column].tolist()
        embeddings = []
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc='Encoding embeddings'):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model_.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        return np.vstack(embeddings)


def build_pipeline(clinical_columns, text_column, embedding_column):
    return FeatureUnion([
        ('clinical', ClinicalPreprocessor(clinical_columns=clinical_columns)),
        ('words', WordFeatureGenerator(text_column=text_column)),
        ('embeddings', EmbeddingTransformer(text_column=embedding_column))
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