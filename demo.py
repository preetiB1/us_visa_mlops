from dotenv import load_dotenv
import os

load_dotenv()
from us_visa.pipline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()
