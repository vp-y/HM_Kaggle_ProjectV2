# Kaggle configuration for the H&M purchase prediction project.
# Set DATA_DIR to the Kaggle dataset directory.
DATA_DIR = "/kaggle/input/h-and-m-personalized-fashion-recommendations"

# FAST_MODE=True is recommended for the first successful Kaggle run.
# Set False for a fuller model comparison/tuning run.
FAST_MODE = True

RANDOM_STATE = 42
HORIZON_DAYS = 30

# Models trained in the clean Kaggle runner.
# Logistic Regression is the baseline. Tree/boosting models are compared against it.
RUN_MODELS = ["logistic", "decision_tree", "random_forest", "extra_trees", "gradient_boosting", "xgboost"]

# Maximum customers used for model fitting in FAST_MODE.
# Feature engineering still uses the transaction history.
MAX_MODEL_ROWS_FAST = 300_000

# Hyperparameter tuning is optional in the compact Kaggle runner.
RUN_TUNING = False
