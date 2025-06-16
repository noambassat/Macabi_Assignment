import pandas as pd
from src.feature_engineering import generate_features
from src.pipeline import build_pipeline
from src.modeling import train_model, evaluate_model, plot_precision_recall_curve, save_model

from config import Config


def main():
    # Load data
    df = pd.read_csv(Config.RAW_DATA_PATH)

    # Prepare data and split
    X_train, X_test, y_train, y_test = generate_features(df)

    # Build pipeline
    pipeline = build_pipeline(
        clinical_columns=None,
        text_column='clinical_sheet',
        embedding_column='last_week_paragraph'
    )

    # Train model
    model = train_model(pipeline, X_train, y_train)

    # Evaluate
    proba, y_pred = evaluate_model(model, X_test, y_test, threshold=0.5)
    plot_precision_recall_curve(y_test, proba)

    # Save model
    save_model(model, Config.MODEL_PATH)


if __name__ == '__main__':
    main()