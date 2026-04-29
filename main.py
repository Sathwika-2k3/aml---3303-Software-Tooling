"""
main.py — Airbnb NYC 2019 Price Prediction Pipeline

This script runs the complete end-to-end ML pipeline:
  1. Loads Airbnb NYC 2019 data from a public URL
  2. Preprocesses the data
  3. Trains 3 models: Linear Regression, Ridge Regression, Random Forest
  4. Logs all params, metrics, and models to MLflow
  5. Registers the best model in MLflow Model Registry

Usage:
    python main.py

Then view results:
    mlflow ui        → open http://localhost:5000
"""

import logging
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.dataloader import load_data, AIRBNB_DATA_URL
from src.preprocessor import preprocess, split_features_target

# ── Settings ──────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Airbnb_Pricing"
REGISTRY_NAME   = "Best_Airbnb_Model_RandomForest"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred: dict):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {"mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def train_and_log(model, run_name: str, params: dict,
                  X_train, X_test, y_train, y_test):
    """Train a model and log everything to MLflow. Returns (model, metrics, run_id)."""
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"{run_name:25s} | MSE={metrics['mse']:>10.2f} | "
                    f"RMSE={metrics['rmse']:>8.2f} | R²={metrics['r2']:.4f}")
        return model, metrics, run_id


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  Airbnb Price Prediction Pipeline — Starting")
    logger.info("=" * 60)

    # 1. Load data
    df_raw = load_data(AIRBNB_DATA_URL)

    # 2. Preprocess
    df_clean = preprocess(df_raw)
    X, y     = split_features_target(df_clean, target_col="price")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 4. MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"\nLogging runs to MLflow experiment: '{EXPERIMENT_NAME}'")
    logger.info("-" * 60)

    results = {}

    # Linear Regression
    _, lr_metrics, lr_run_id = train_and_log(
        model     = LinearRegression(),
        run_name  = "Linear_Regression",
        params    = {"model_type": "LinearRegression", "test_size": TEST_SIZE},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    results["Linear_Regression"] = {"metrics": lr_metrics, "run_id": lr_run_id}

    # Ridge Regression
    ALPHA = 1.0
    _, ridge_metrics, ridge_run_id = train_and_log(
        model     = Ridge(alpha=ALPHA),
        run_name  = "Ridge_Regression",
        params    = {"model_type": "Ridge", "alpha": ALPHA, "test_size": TEST_SIZE},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    results["Ridge_Regression"] = {"metrics": ridge_metrics, "run_id": ridge_run_id}

    # Random Forest
    N_ESTIMATORS = 100
    _, rf_metrics, rf_run_id = train_and_log(
        model     = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
        run_name  = "Random_Forest",
        params    = {"model_type": "RandomForest", "n_estimators": N_ESTIMATORS,
                     "random_state": RANDOM_STATE, "test_size": TEST_SIZE},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    results["Random_Forest"] = {"metrics": rf_metrics, "run_id": rf_run_id}

    # 5. Select best model (lowest MSE)
    best_name   = min(results, key=lambda k: results[k]["metrics"]["mse"])
    best_run_id = results[best_name]["run_id"]

    logger.info("-" * 60)
    logger.info(f"Best model: {best_name}  (MSE = {results[best_name]['metrics']['mse']})")

    # 6. Register best model
    model_uri  = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
    logger.info(f"Registered '{REGISTRY_NAME}' — version {registered.version}")

    logger.info("=" * 60)
    logger.info("  Pipeline Complete!  Run `mlflow ui` to view results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
