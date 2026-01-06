import json
import sys
import pandas as pd
from pandas import DataFrame

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, df: DataFrame) -> bool:
        try:
            required_columns = set(self.schema_config["columns"].keys())
            present_columns = protect = set(df.columns)

            status = required_columns.issubset(present_columns)
            logging.info(f"Column count validation status: {status}")

            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_column_existence(self, df: DataFrame) -> bool:
        try:
            missing_columns = []

            for column in (
                self.schema_config["numerical_columns"]
                + self.schema_config["categorical_columns"]
            ):
                if column not in df.columns:
                    missing_columns.append(column)

            if missing_columns:
                logging.error(f"Missing columns: {missing_columns}")
                return False

            return True
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(
        self, reference_df: DataFrame, current_df: DataFrame
    ) -> bool:
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(reference_df, current_df)

            report = json.loads(profile.json())
            write_yaml_file(
                self.data_validation_config.drift_report_file_path, report
            )

            metrics = report["data_drift"]["data"]["metrics"]
            logging.info(
                f"Drift detected: {metrics['n_drifted_features']} / {metrics['n_features']}"
            )

            return metrics["dataset_drift"]
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            validation_error = ""

            if not self.validate_number_of_columns(train_df):
                validation_error += "Missing columns in training data. "

            if not self.validate_number_of_columns(test_df):
                validation_error += "Missing columns in testing data. "

            if not self.validate_column_existence(train_df):
                validation_error += "Invalid columns in training data. "

            if not self.validate_column_existence(test_df):
                validation_error += "Invalid columns in testing data. "

            validation_status = validation_error == ""

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                message = (
                    "Drift detected" if drift_status else "No drift detected"
                )
            else:
                message = validation_error

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data Validation Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys)
