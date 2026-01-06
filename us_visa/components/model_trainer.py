import sys
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # ---------------------------------------------------------
    # Baseline models
    # ---------------------------------------------------------
    def _get_baseline_models(self):
        return {
            "DecisionTree": DecisionTreeClassifier(
                class_weight="balanced", random_state=1
            ),
            "Bagging": BaggingClassifier(random_state=1),
            "RandomForest": RandomForestClassifier(
                class_weight="balanced", random_state=1
            ),
            "AdaBoost": AdaBoostClassifier(random_state=1),
            "GradientBoosting": GradientBoostingClassifier(random_state=1),
            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=1,
            ),
        }

    # ---------------------------------------------------------
    # Hyperparameter grids (ONLY for tuned models)
    # ---------------------------------------------------------
    def _get_param_grids(self):
        return {
            "RandomForest": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 3, 5],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        }

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    def _evaluate(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "train_recall": recall_score(y_train, y_train_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "test_f1": f1_score(y_test, y_test_pred),
        }

    # ---------------------------------------------------------
    # Hyperparameter tuning + MLflow logging
    # ---------------------------------------------------------
    def _tune_and_log(
        self,
        model_name,
        model,
        param_grid,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        with mlflow.start_run(run_name=f"{model_name}_Tuned"):
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1,
            )

            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            metrics = self._evaluate(
                best_model, X_train, y_train, X_test, y_test
            )

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")

            return best_model, metrics

    # ---------------------------------------------------------
    # Pipeline entry point
    # ---------------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            mlflow.set_experiment("US_Visa_Models")

            # ---------------- BASELINE MODELS ----------------
            baseline_models = self._get_baseline_models()
            baseline_results = {}

            for name, model in baseline_models.items():
                with mlflow.start_run(run_name=name):
                    metrics = self._evaluate(
                        model, X_train, y_train, X_test, y_test
                    )
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, "model")
                    baseline_results[name] = metrics

            # ---------------- HYPERPARAMETER TUNING ----------------
            param_grids = self._get_param_grids()
            tuned_results = {}

            for name in param_grids:
                model = baseline_models[name]
                tuned_model, metrics = self._tune_and_log(
                    name,
                    model,
                    param_grids[name],
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
                tuned_results[name] = (tuned_model, metrics)

            # ---------------- FINAL MODEL SELECTION ----------------
            best_model_name = max(
                tuned_results,
                key=lambda k: tuned_results[k][1]["test_f1"],
            )

            final_model, final_metrics = tuned_results[best_model_name]

            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_model,
            )

            metric_artifact = ClassificationMetricArtifact(
                f1_score=final_metrics["test_f1"],
                precision_score=final_metrics["test_precision"],
                recall_score=final_metrics["test_recall"],
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

        except Exception as e:
            raise USvisaException(e, sys) from e
