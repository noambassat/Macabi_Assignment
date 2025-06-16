import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import LGBMModel

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import lightgbm as lgb


from sklearn.pipeline import Pipeline


def train_model(feature_pipeline, X_train, y_train):
    """
    Creates full pipeline (features + model) and fits it.
    """
    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', LGBMModel().get_model())
    ])
    full_pipeline.fit(X_train, y_train)
    return full_pipeline



def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates model performance with classification report and confusion matrix.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    print(f"Classification Report (Threshold={threshold:.2f}):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix (Threshold={threshold:.2f})')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

    return proba, y_pred


def plot_precision_recall_curve(y_true, y_proba):
    """
    Plots precision-recall tradeoff vs. thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]
    diff = np.abs(precision - recall)
    best_idx = np.argmin(diff)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label='Precision', linewidth=2)
    plt.plot(thresholds, recall, label='Recall', linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold = {best_threshold:.2f}')
    plt.scatter(best_threshold, best_precision, color='red', s=100)
    plt.text(best_threshold + 0.01, best_precision - 0.05,
             f'Prec={best_precision:.2f}\nRec={best_recall:.2f}',
             fontsize=12, color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs. Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_model(model, path):
    """
    Saves the trained model pipeline to disk.
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Loads a trained model pipeline from disk.
    """
    return joblib.load(path)