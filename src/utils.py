import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def save_artifact(obj, path):
    """
    Saves any Python object to disk using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path):
    """
    Loads a joblib object from disk.
    """
    return joblib.load(path)


def save_plot(fig, path):
    """
    Saves a matplotlib figure to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(cm, title, labels):
    """
    Plots a labeled confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()
    return fig
