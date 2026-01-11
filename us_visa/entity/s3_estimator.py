from us_visa.cloud_storage.aws_storage import SimpleStorageService


class USvisaEstimator:

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()

    def save_model(self, model_path: str, s3_key: str):
        """
        Uploads a trained model file to S3

        model_path : local file path of model.pkl
        s3_key     : key inside bucket (e.g. 'model.pkl' or 'models/usvisa/model.pkl')
        """

        self.s3.upload_file(
            from_filename=model_path,
            to_filename=s3_key,
            bucket_name=self.bucket_name
        )

    def is_model_present(self, s3_key: str) -> bool:
        return self.s3.s3_key_path_available(
            bucket_name=self.bucket_name,
            s3_key=s3_key
        )

    def load_model(self, s3_key: str, local_path: str):
        self.s3.load_model(
            model_path=s3_key,
            bucket_name=self.bucket_name
        )
