import sys
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
)
from us_visa.entity.s3_estimator import USvisaEstimator


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    # --------------------------------------------------------
    # Get production model from S3 (if exists)
    # --------------------------------------------------------
    def get_best_model(self):
     try:
        model_key = self.model_eval_config.s3_model_key_path

        estimator = USvisaEstimator(
            bucket_name=self.model_eval_config.bucket_name
        )

        if estimator.is_model_present(s3_key=model_key):
            return estimator
        else:
            return None

     except Exception as e:
        raise USvisaException(e, sys)


    # --------------------------------------------------------
    # Compare new model vs production model
    # --------------------------------------------------------
    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            logging.info("Starting model evaluation")

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]

            X = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            # Newly trained model score (already computed during training)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            # Load production model from S3 if exists
            best_model_f1_score = None
            best_model_estimator = self.get_best_model()

            if best_model_estimator is not None:
                best_model = best_model_estimator.load_model()
                y_pred_best = best_model.predict(X)
                best_model_f1_score = f1_score(y, y_pred_best)

            tmp_best_score = 0 if best_model_f1_score is None else best_model_f1_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_score,
                difference=trained_model_f1_score - tmp_best_score,
            )

            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys) from e

    # --------------------------------------------------------
    # Final artifact
    # --------------------------------------------------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluation = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
