import json
import pickle
import boto3
import mlflow
import joblib
import tempfile

import numpy as np
import pandas as pd

# from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

mlflow_server = "http://mlflow:5000"
dtc_model_params = ("dtc_model_prod", "champion", "model_dtc.pkl")
lda_model_params = ("lda_model_prod", "champion", "model_lda.pkl")

def load_model(model_params: tuple[str, str, str]):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. 

    :param model_params[0]: The name of the model.
    :param model_params[1]: The alias of the model version.
    :param model_params[2]: Local filename to load in case of remote error.
    :return: A tuple containing the loaded model and its version.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri(mlflow_server)
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_params[0], model_params[1])
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open(f'/app/files/{model_params[2]}', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    return model_ml, version_model_ml

def load_data_transformation(fallback_columns_file: str, fallback_pipeline_file: str):
    """
    Load columns and categories file and load pipeline transformation file about the ETL pipeline
    from an S3 bucket. 
    If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param fallback_columns_file: Local json filename to load with columns and categories in case of remote error.
    :param fallback_pipeline_file: Local joblib filename to load with pipeline transformation in case of remote error.
    :return: dict_data, pipeline.
    """

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='chicago/crimes/data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='chicago/crimes/data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open(f"/app/files/{fallback_columns_file}", 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    try:
        # Load information of the ETL pipeline from S3
        with tempfile.TemporaryFile() as fp:
            s3 = boto3.client('s3')
            s3.download_fileobj(Bucket='data', Key='chicago/crimes/data_info/pipeline.joblib', Fileobj=fp)
            fp.seek(0)  # Rewind to the beginning of the file-like object
            pipeline = joblib.load(fp)

    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open(f"/app/files/{fallback_pipeline_file}", 'rb')
        pipeline = joblib.load(file_s3)
        file_s3.close()

    return data_dictionary, pipeline

def check_models_update():
    """
    Check for updates in the models and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model_dtc
    global model_lda

    try:
        new_model_dtc = load_model(dtc_model_params)
        # If the versions are not the same
        if model_dtc[1] != new_model_dtc[1]:
            # Replace model with newest
            model_dtc = new_model_dtc
    except:
        # If an error occurs during the process, pass silently
        pass

    try:
        new_model_lda = load_model(lda_model_params)
        # If the versions are not the same
        if model_lda[1] != new_model_lda[1]:
            # Replace model with newest
            model_lda = new_model_lda
    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the Chicago Crimes prediction model.

    This class defines the input fields required by the Chicago Crimes prediction model along with their descriptions
    and validation constraints.

    :param iucr: The Illinois Uniform Crime Reporting code.
    :param primary_type: The primary description of the IUCR code.
    :param description: The secondary description of the IUCR code, a subcategory of the primary description.
    :param location_description: Description of the location where the incident occurred.
    :param arrest: Indicates whether an arrest was made.
    :param domestic: Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act.
    :param beat: Indicates the beat where the incident occurred.
    :param district: Indicates the police district where the incident occurred.
    :param ward: The ward (City Council district) where the incident occurred.
    :param community_area: Indicates the community area where the incident occurred.
    :param latitude: The latitude of the location where the incident occurred.
    :param longitude: The longitude of the location where the incident occurred.
    :param mes: Month in witch the incident ocurred.
    :param dia_mes: Day of month in witch the incident ocurred.
    :param dia_semana: Day of week in witch the incident ocurred.
    :param hora: Hour in witch the incident ocurred.
    """

    iucr: str = Field(
        description="The Illinois Uniform Crime Reporting code.",
        min_length=4,
        max_length=4,
    )
    primary_type: str = Field(
        description="The primary description of the IUCR code.",
        min_length=1
    )
    description: str = Field(
        description="The secondary description of the IUCR code, a subcategory of the primary description.",
        min_length=1
    )
    location_description: str = Field(
        description="Description of the location where the incident occurred.",
        min_length=1
    )
    arrest: bool = Field(
        description="Indicates whether an arrest was made."
    )
    domestic: bool = Field(
        description="Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act."
    )
    beat: int = Field(
        description="Indicates the beat where the incident occurred.",
        ge=1,
        le=9999,
    )
    district: int = Field(
        description="Indicates the police district where the incident occurred.",
        ge=1,
        le=999,
    )
    ward: int = Field(
        description="The ward (City Council district) where the incident occurred.",
        ge=1,
        le=99,
    )
    community_area: int = Field(
        description="Indicates the community area where the incident occurred.",
        ge=1,
        le=99,
    )
    latitude: float = Field(
        description="The latitude of the location where the incident occurred.",
        ge=-90,
        le=90,
    )
    longitude: float = Field(
        description="The longitude of the location where the incident occurred.",
        ge=-180,
        le=180,
    )
    mes: int = Field(
        description="Month in witch the incident ocurred.",
        ge=1,
        le=12,        
    )
    dia_mes: int = Field(
        description="Day of month in witch the incident ocurred.",
        ge=1,
        le=31, 
    )
    dia_semana: int = Field(
        description="Day of week in witch the incident ocurred.",
        ge=0,
        le=6, 
    )
    hora: int = Field(
        description="Hour in witch the incident ocurred.",
        ge=0,
        le=23, 
    )

    model_config = {
        "json_schema_extra": {
            "examples": 
                        [
                            {
                                "iucr": "0930",
                                "primary_type": "MOTOR VEHICLE THEFT",
                                "description": "THEFT / RECOVERY - AUTOMOBILE",
                                "location_description": "STREET",
                                "arrest": False,
                                "domestic": False,
                                "beat": 412,
                                "district": 4,
                                "ward": 7,
                                "community_area": 46,
                                "latitude": 41.743747398,
                                "longitude": -87.566115082,
                                "mes": 6,
                                "dia_mes": 4,
                                "dia_semana": 1,
                                "hora": 5
                            },
                            {
                                "iucr": "0910",
                                "primary_type": "MOTOR VEHICLE THEFT",
                                "description": "AUTOMOBILE",
                                "location_description": "STREET",
                                "arrest": False,
                                "domestic": False,
                                "beat": 1233,
                                "district": 12,
                                "ward": 25,
                                "community_area": 31,
                                "latitude": 41.856086444,
                                "longitude": -87.659628733,
                                "mes": 6,
                                "dia_mes": 17,
                                "dia_semana": 0,
                                "hora": 23
                            },
                            {
                                "iucr": "1477",
                                "primary_type": "WEAPONS VIOLATION",
                                "description": "RECKLESS FIREARM DISCHARGE",
                                "location_description": "RESIDENCE",
                                "arrest": False,
                                "domestic": False,
                                "beat": 2512,
                                "district": 25,
                                "ward": 29,
                                "community_area": 18,
                                "latitude": 41.927457482,
                                "longitude": -87.806374185,
                                "mes": 6,
                                "dia_mes": 13,
                                "dia_semana": 3,
                                "hora": 2
                            },
                            {
                                "iucr": "1310",
                                "primary_type": "CRIMINAL DAMAGE",
                                "description": "TO PROPERTY",
                                "location_description": "APARTMENT",
                                "arrest": False,
                                "domestic": False,
                                "beat": 413,
                                "district": 4,
                                "ward": 8,
                                "community_area": 47,
                                "latitude": 41.727193361,
                                "longitude": -87.600963809,
                                "mes": 2,
                                "dia_mes": 12,
                                "dia_semana": 0,
                                "hora": 2
                            },
                            {
                                "iucr": "0610",
                                "primary_type": "BURGLARY",
                                "description": "FORCIBLE ENTRY",
                                "location_description": "APARTMENT",
                                "arrest": False,
                                "domestic": False,
                                "beat": 1932,
                                "district": 19,
                                "ward": 32,
                                "community_area": 7,
                                "latitude": 41.926133022,
                                "longitude": -87.663357067,
                                "mes": 10,
                                "dia_mes": 4,
                                "dia_semana": 4,
                                "hora": 16
                            }
                        ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the Chicago Crimes prediction model.

    This class defines the output fields returned by the Chicago Crimes prediction model along with their descriptions
    and possible values.

    :param dtc_fbi_code_output: Output of the model. FBI Code predicted with Decision Tree Classifier.
    :param lda_fbi_code_output: Output of the model. FBI Code predicted Linear Discriminant Analysis.
    """

    dtc_fbi_code_output: str = Field(
        description="Output of the model predicted with Decision Tree Classifier. FBI Code",
    )
    lda_fbi_code_output: str = Field(
        description="Output of the model predicted with Linear Discriminant Analysis. FBI Code",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dtc_fbi_code_output": "08B",
                    "lda_fbi_code_output": "08B",
                }
            ]
        }
    }


# Load the model before start
model_dtc = load_model(dtc_model_params)
model_lda = load_model(lda_model_params)
data_dict, pipeline = load_data_transformation("data.json", "pipeline.joblib")

app = FastAPI()

@app.get("/")
async def read_root():
    """
    Root endpoint of the Chicago Crime to FBI Code API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Chicago Crime to FBI Code API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting FBI Code from Chicago Crime Record.

    This endpoint receives features related to Chicago Police Record and predicts the FBI Code using a two trained models.
    It returns the prediction result in string format.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.model_dump().values()]
    features_key = [*features.model_dump().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Process categorical features
    for categorical_col in data_dict["categorical_columns"]:
        # features_df[categorical_col] = features_df[categorical_col].astype(int)
        categories = data_dict["categories_values_per_categorical"][categorical_col]
        features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)

    # Reorder DataFrame columns
    features_df = features_df[data_dict["columns"]]

    # Transform features with prefitted pipeline
    features_df = pipeline.transform(features_df)

    # Make the prediction using the trained models
    prediction_dtc = model_dtc[0].predict(features_df)
    prediction_lda = model_lda[0].predict(features_df)

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_models_update)

    # Return the prediction result
    return ModelOutput(dtc_fbi_code_output=prediction_dtc[0], lda_fbi_code_output=prediction_lda[0])
