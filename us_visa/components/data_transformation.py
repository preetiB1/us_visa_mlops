import sys
import numpy as np
import pandas as pd

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_numpy_array_data, read_yaml_file, drop_columns


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # ----------------- Split input & target -----------------
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # ----------------- Drop unwanted columns -----------------
            drop_cols = self.schema_config["drop_columns"]

            X_train = drop_columns(X_train, drop_cols)
            X_test = drop_columns(X_test, drop_cols)

            # ----------------- Encode target (NOTEBOOK LOGIC) -----------------
            y_train = y_train.apply(lambda x: 1 if x == "Certified" else 0)
            y_test = y_test.apply(lambda x: 1 if x == "Certified" else 0)

            # ----------------- One-hot encoding -----------------
            X_train = pd.get_dummies(X_train, drop_first=True)
            X_test = pd.get_dummies(X_test, drop_first=True)

            # Align test columns with train columns
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            # ----------------- Create final arrays -----------------
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)

            train_arr = np.c_[X_train.values, y_train.values]
            test_arr = np.c_[X_test.values, y_test.values]


            # ----------------- Save outputs -----------------
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            logging.info("Data transformation completed successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise USvisaException(e, sys) from e
