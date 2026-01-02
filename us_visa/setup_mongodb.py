import os
import sys
import pandas as pd
import pymongo
import json
import certifi

# If these imports fail, make sure your us_visa/constants/__init__.py is set up!
from us_visa.constants import DATABASE_NAME, COLLECTION_NAME, MONGODB_URL_KEY

def push_data_to_mongodb(csv_file_path: str):
    try:
        # 1. Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"File not found at: {csv_file_path}")

        print(f"Reading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # 2. Convert Dataframe to JSON format (list of dictionaries)
        print("Converting data to JSON...")
        data_json = json.loads(df.to_json(orient='records'))

        # 3. Get MongoDB URL from Environment Variable
        mongo_db_url = os.getenv(MONGODB_URL_KEY)
        if mongo_db_url is None:
            raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")

        # 4. Connect to MongoDB
        # ca = certifi.where() is used to avoid SSL errors on some systems
        print("Connecting to MongoDB...")
        ca = certifi.where()
        client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
        
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # 5. Insert Data
        print(f"Inserting {len(data_json)} records into {DATABASE_NAME}.{COLLECTION_NAME}...")
        collection.insert_many(data_json)
        
        print("Data pushed successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Ensure 'Visadataset.csv' is in the same folder as this script
    FILE_PATH = "EasyVisa.csv" 
    push_data_to_mongodb(FILE_PATH)