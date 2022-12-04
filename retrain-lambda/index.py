from sagemaker.mxnet import MXNet
from sagemaker.predictor import Predictor

import json
import os

bucket_name = os.environ['SAGEMAKER_BUCKET']
bucket_key_prefix = os.environ['SAGEMAKER_BUCKET_KEY_PREFIX']
role = os.environ['SAGEMAKER_ROLE']
prediction_endpoint = os.environ['PREDICTION_ENDPOINT']


def lambda_handler(event, context):
    retrain()
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Inference Lambda!')
    }


def retrain():
    output_path = 's3://{0}/{1}/output'.format(bucket_name, bucket_key_prefix)
    code_location = 's3://{0}/{1}/code'.format(bucket_name, bucket_key_prefix)

    m = MXNet('sms_spam_classifier_mxnet_script.py',
              role=role,
              instance_count=1,
              instance_type='ml.c5.2xlarge',
              output_path=output_path,
              base_job_name='sms-spam-classifier-mxnet',
              framework_version='1.2',
              py_version='py3',
              code_location=code_location,
              hyperparameters={
                  'batch_size': 100,
                  'epochs': 20,
                  'learning_rate': 0.01
              })

    inputs = {
        'train': 's3://{0}/{1}/train/'.format(bucket_name, bucket_key_prefix),
        'val': 's3://{0}/{1}/val/'.format(bucket_name, bucket_key_prefix)
    }

    m.fit(inputs)

    model = m.create_model()
    model.create(instance_type='ml.c5.2xlarge')

    mxnet_pred = Predictor(endpoint_name=prediction_endpoint)
    mxnet_pred.update_endpoint(initial_instance_count=1,
                               instance_type='ml.c5.2xlarge',
                               model_name=model.name)
