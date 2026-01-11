import os
import sys
from boto3.s3.transfer import TransferConfig
from us_visa.configuration.aws_connection import S3Client
from us_visa.exception import USvisaException
from us_visa.logger import logging


class SimpleStorageService:

    def __init__(self):
        try:
            s3 = S3Client()
            self.s3_client = s3.s3_client
            self.s3_resource = s3.s3_resource

            # CRITICAL: disable multipart uploads on Windows
            self.transfer_config = TransferConfig(
                multipart_threshold=1024 * 1024 * 1024,  # 1GB
                max_concurrency=1,
                use_threads=False
            )

        except Exception as e:
            raise USvisaException(e, sys)

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            return True
        except:
            return False

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove=False):
        try:
            logging.info(f"Uploading {from_filename} → s3://{bucket_name}/{to_filename}")

            self.s3_client.upload_file(
                Filename=from_filename,
                Bucket=bucket_name,
                Key=to_filename,
                Config=self.transfer_config
            )

            if remove:
                os.remove(from_filename)

            logging.info("Upload successful")

        except Exception as e:
            raise USvisaException(e, sys)
