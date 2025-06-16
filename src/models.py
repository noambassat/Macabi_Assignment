# src/models.py
import lightgbm as lgb

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
