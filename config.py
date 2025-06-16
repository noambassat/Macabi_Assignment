class Config:
    RAW_DATA_PATH = "data/ds_assignment_data.csv"
    MODEL_PATH = "models/final_model.joblib"
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    TEXT_COLUMN = "clinical_sheet"
    EMBEDDING_COLUMN = "last_week_paragraph"
    TARGET_COLUMN = "Y"
