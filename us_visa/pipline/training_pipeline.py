import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer

from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Completed data ingestion")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            artifact = data_validation.initiate_data_validation()

            logging.info("Completed data validation")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info("Completed data transformation")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_model_training(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            artifact = model_trainer.initiate_model_trainer()

            logging.info("Completed model training")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                ingestion_artifact, validation_artifact
            )

            self.start_model_training(transformation_artifact)

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            raise USvisaException(e, sys)
