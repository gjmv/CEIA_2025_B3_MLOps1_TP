import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Chicago Crimes Dataset

Este DAG extrae información del dataset original de Crímenes de Chicago del año 2024, ubicado en Chicago Data Portal.
[Crímenes Chicago 2024](https://data.cityofchicago.org/Public-Safety/Crimes-2024/dqcy-ctma/about_data). 

It preprocesses the data by creating dummy variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Mealla, Mendoza, Viñas",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="process_etl_chicago_crimes_2024",
    description="ETL process for Chicago Crimes, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Chicago Crimes", "2024"],
    default_args=default_args,
    catchup=False,
)
def process_etl_chicago_crimes_2024():
    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["sodapy==2.2.0",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_data(s3_base_path):
        """
        Load the raw data from Chicago Data Portal and save in S3
        """
        import pandas as pd
        import awswrangler as wr
        from sodapy import Socrata


        entity_set = "dqcy-ctma" # Crímenes 2024
        dataframe = None

        client = Socrata("data.cityofchicago.org", None)
        offset = 0
        limit=2000
        while True:
            results = client.get(entity_set, limit=limit, offset=offset)
            if (len(results) == 0):
                break

            # Convert to pandas DataFrame
            if (dataframe is None):
                dataframe = pd.DataFrame.from_records(results)
            else:
                dataframe = pd.concat([dataframe, pd.DataFrame.from_records(results)], ignore_index=True)
            offset += limit

        if dataframe is None:
            raise(Exception("No data downloaded from CDP."))
        
        data_path = f"{s3_base_path}/chicago_crimes_2024.csv"
        wr.s3.to_csv(df=dataframe,
                     path=data_path,
                     index=False)

    @task.virtualenv(
        task_id="pre_process_columns",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def pre_process_columns(s3_base_path):
        """
        Pre-process variables, drop duplicates, convert columns, etc.
        """
        import awswrangler as wr
        import pandas as pd

        data_original_path = f"{s3_base_path}/chicago_crimes_2024.csv"
        data_preprocessed_path = f"{s3_base_path}/chicago_crimes_2024_preprocessed.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # Analizamos filas duplicadas, sin considerar las siguientes columnas
        drop_columns = ["id", "date", "updated_on"]
        df_dup = dataset.drop(drop_columns, axis=1)

        # Ordenamos por fecha de crimen y de actualización, para que al momento de eliminar duplicados, podamos quedarnos con el último
        # Eliminamos los duplicados obtenidos, consideramos los iguales ignorando la fecha del suceso, la actualización, y el id del dataset (que no es el id del suceso).
        dataset.sort_values(axis=0, by=["updated_on", "date"], ascending=True).drop_duplicates(df_dup.columns, keep='last', inplace=True)

        # Convertimos columnas del dataset al tipo correcto y agregamos datos extraidos de la fecha
        dataset["date"] = pd.to_datetime(dataset["date"], format="%Y-%m-%dT%H:%M:%S.%f")
        dataset["mes"] = dataset["date"].dt.month
        dataset["dia_mes"] = dataset["date"].dt.day
        dataset["dia_semana"] = dataset["date"].dt.dayofweek
        dataset["hora"] = dataset["date"].dt.hour

        # Eliminamos columnas que consideramos no tienen aporte significativo, o que tiene valores únicos para cada fila (ids)
        drop_columns = ["id", "date", "case_number", "block", "x_coordinate", "y_coordinate", "year", "updated_on", "location"]
        dataset.drop(drop_columns, inplace=True, axis=1)
        dataset.dropna(inplace=True)

        categorical_cols = ['iucr', 'primary_type', 'description', 'location_description', 'beat', 'district', 'ward', 'community_area', 'fbi_code']
        dataset[categorical_cols] = dataset[categorical_cols].astype('category')

        wr.s3.to_csv(df=dataset,
                     path=data_preprocessed_path,
                     index=False)

        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow
        import numpy as np

        # import awswrangler as wr
        # import pandas as pd

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='chicago/crimes/data_info/data.json')
            result = client.get_object(Bucket='data', Key='chicago/crimes/data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        target_col = "fbi_code"

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset.columns.drop("fbi_code").to_list()
        data_dict['target_col'] = "fbi_code"
        data_dict['categorical_columns'] = categorical_cols

        categories_dict = {}
        for category in categorical_cols:
            categories_dict[category] = np.sort(dataset[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = categories_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(Bucket='data', Key='chicago/crimes/data_info/data.json', Body=data_string)

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Chicago Crimes 2024")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Chicago Crimes", "year": "2024"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(dataset,
                                                 source="https://data.cityofchicago.org/Public-Safety/Crimes-2024/dqcy-ctma/about_data",
                                                 targets=target_col,
                                                 name="chicago_crimes_2024")
        mlflow.log_input(mlflow_dataset, context="Dataset")

    @task.virtualenv(
        task_id="dataset_pre_process_data",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.7.1",
                      "category_encoders==2.8.1"],
        system_site_packages=True
    )
    def dataset_pre_process_data(s3_base_path):
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler
        from sklearn.impute import SimpleImputer
        from category_encoders import TargetEncoder
        from sklearn.compose import ColumnTransformer
        import pandas as pd
        import mlflow
        import boto3
        import joblib
        import tempfile

        data_preprocessed_path = f"{s3_base_path}/chicago_crimes_2024_preprocessed.csv"
        dataset = wr.s3.read_csv(data_preprocessed_path)

        test_size = 0.3
        target_col = "fbi_code"

        # Separamos los features y el target
        X = dataset.drop(columns=target_col)
        y = dataset[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        numericas = ["latitude", "longitude"]
        cardinalidad_baja = ["arrest", "domestic", "dia_semana"]
        cardinalidad_media = ["iucr", "primary_type", "description", "location_description", "beat", "district", "ward", "community_area", "mes", "dia_mes", "hora"]
        
        # Convierte el dato en string
        def to_str(X):
            return X.astype(str)

        transformar_numericas = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        transformar_bajacard = Pipeline(steps=[
            ('to_str', FunctionTransformer(to_str, validate=False)), # Para que no falle con categoricas string
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # drop=first para reducir los features
        ])

        transformar_nominales = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encoder', TargetEncoder(handle_unknown='value')), # Si se encuentra con una categoría desconocida, codifica con la media global del set de entrenamiento
            ('scaler', MinMaxScaler()),
        ])
        
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', transformar_numericas, numericas),
                ('bin', transformar_bajacard, cardinalidad_baja),
                ('nom', transformar_nominales, cardinalidad_media)
            ],
            remainder='drop'
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        # Hacemos fit y transform sobre train
        X_train_procesado = pipeline.fit_transform(X_train, y_train)
        # Ahora aplicamos el transform en test
        X_test_procesado = pipeline.transform(X_test)
        

        numericas_salida = numericas # las numéricas transformadas
        nominales_salida = cardinalidad_media # las categoricas de cardinalidad media transformadas
        binarias_salida = pipeline.named_steps['preprocessor'].named_transformers_['bin'].named_steps['onehot'].get_feature_names_out(cardinalidad_baja)

        # Armamos los nombres de las variables en orden de acuerdo a la pipeline
        transformed_feature_names = (
            list(numericas_salida) +
            list(binarias_salida) +
            list(nominales_salida) 
        )

        # Reconstruimos el dataframe procesado
        X_train_preproc = pd.DataFrame(X_train_procesado, columns=transformed_feature_names)
        X_test_preproc = pd.DataFrame(X_test_procesado, columns=transformed_feature_names)

        wr.s3.to_csv(df=pd.DataFrame(X_train_preproc), path=f"{s3_base_path}/final/X_train.csv", index=False)
        wr.s3.to_csv(df=pd.DataFrame(X_test_preproc), path=f"{s3_base_path}/final/X_test.csv", index=False)
        wr.s3.to_csv(df=pd.DataFrame(y_train), path=f"{s3_base_path}/final/y_train.csv", index=False)
        wr.s3.to_csv(df=pd.DataFrame(y_test), path=f"{s3_base_path}/final/y_test.csv", index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        with tempfile.TemporaryFile() as fp:
                joblib.dump(pipeline, fp)
                fp.seek(0)  # Rewind to the beginning of the file-like object
                client.put_object(Bucket="data", Key='chicago/crimes/data_info/pipeline.joblib', Body=fp.read())

    s3_base_path = "s3://data/chicago/crimes/2024"
    get_data(s3_base_path) >> pre_process_columns(s3_base_path) >> dataset_pre_process_data(s3_base_path)


dag = process_etl_chicago_crimes_2024()