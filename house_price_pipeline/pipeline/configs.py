"""TFX House Pricing Pipeline Config.

This file defines environments for a TFX taxi pipeline.
"""

import os


PIPELINE_NAME = 'house_price_pipeline'

# Retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
  import google.auth 
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = ''
except ImportError:
  GOOGLE_CLOUD_PROJECT = ''


# Specify your GCS bucket name here. You have to use GCS to store output files
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.model.run_fn'

TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 150

EVAL_ACCURACY_THRESHOLD = 0.8



