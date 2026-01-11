import sys
from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.s3_estimator import USvisaEstimator


class ModelPusher:
    def __init__(self,
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):

        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.usvisa_estimator = USvisaEstimator(
            bucket_name=model_pusher_config.bucket_name
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:

        try:
            logging.info("Starting Model Pusher")

            local_model_path = self.model_evaluation_artifact.trained_model_path

            s3_key = self.model_pusher_config.s3_model_key_path   # ex: "model.pkl"

            logging.info(f"Uploading model → s3://{self.model_pusher_config.bucket_name}/{s3_key}")

            self.usvisa_estimator.save_model(
                model_path=local_model_path,
                s3_key=s3_key
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=s3_key
            )

            logging.info("Model upload successful")
            return model_pusher_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
