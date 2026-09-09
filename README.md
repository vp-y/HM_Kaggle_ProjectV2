# H&M Customer Purchase Prediction — Kaggle Run Package

## Objective

Predict whether a customer will make **at least one purchase in the next 30 days** using only information available at a historical cutoff date.

The original H&M Kaggle competition is a recommendation problem. This project deliberately reformulates it as a customer-level supervised binary classification problem so that it can demonstrate a complete core-ML workflow.

**Images are not used.**

## What the Kaggle runner produces

The clean runner trains and compares:

- Logistic Regression — baseline
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost, when installed

It reports:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Train/validation generalization gap
- Model fit time
- Best model
- Best model metrics
- Absolute improvement over baseline
- Relative improvement over baseline

The primary model-selection metric is **validation PR-AUC**, which is more informative than accuracy for an imbalanced purchase target.

## Kaggle dataset

Add the Kaggle dataset:

`H&M Personalized Fashion Recommendations`

The expected files are:

- `transactions_train.csv`
- `customers.csv`
- `articles.csv`

The `images/` directory is ignored.

## Recommended GitHub structure

Push this complete folder to GitHub. In Kaggle:

1. Create a Kaggle Notebook.
2. Add the H&M dataset under **Add Input**.
3. Clone the GitHub repository in the first cell.
4. Run `notebooks/Kaggle_End_to_End_Run.ipynb`, or execute `src/kaggle_runner.py`.
5. Open `results/final_model_comparison.csv` for the main model-wise score table.
6. Open `results/baseline_improvement.csv` for the baseline-vs-best-model improvement.
7. Open `results/best_model.json` for the selected model.

## Important

The runner keeps the final test period untouched during model selection. The operating threshold is selected on validation data and then applied once to the test period.

For a first run, keep `FAST_MODE=True`. After confirming that the pipeline works, set it to `False` and increase the model/tuning budget as needed.
