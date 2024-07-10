# leads

``
export MLFLOW_TRACKING_URI=https://dagshub.com/minich-code/leads.mlflow
export MLFLOW_TRACKING_USERNAME=minich-code
export MLFLOW_TRACKING_PASSWORD=cadc5e14617d7fae5ed8a6532906afca14f3b0f9
``

``
dvc remote add origin s3://dvc
dvc remote modify origin  endpointurl https://dagshub.com/minich-code/leads.s3
dvc remote modify origin --local access_key_id cadc5e14617d7fae5ed8a6532906afca14f3b0f9
dvc remote modify origin --local secret_access_key cadc5e14617d7fae5ed8a6532906afca14f3b0f9
``
