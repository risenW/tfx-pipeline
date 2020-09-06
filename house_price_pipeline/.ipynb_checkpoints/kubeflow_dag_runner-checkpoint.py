"""Define KubeflowDagRunner to run the pipeline using Kubeflow."""



import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils


# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

DATA_PATH = 'gs://{}/tfx-hprice/data/'.format(configs.GCS_BUCKET_NAME)



def run():
  """Define a kubeflow pipeline."""

  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config, tfx_image=tfx_image)
    
  pod_labels = kubeflow_dag_runner.get_default_pod_labels()
  pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: 'tfx-hprice'})
  kubeflow_dag_runner.KubeflowDagRunner(
      config=runner_config, pod_labels_to_attach=pod_labels
  ).run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
      ))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
