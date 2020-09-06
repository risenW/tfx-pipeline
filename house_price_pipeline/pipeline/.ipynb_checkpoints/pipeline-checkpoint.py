"""TFX House Pricing pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""


from ml_metadata.proto import metadata_store_pb2
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component  # pylint: disable=unused-import
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input
import tensorflow_model_analysis as tfma


def create_pipeline(pipeline_name, 
                    pipeline_root, 
                    data_path,
                    preprocessing_fn,
                    run_fn,
                    train_args,
                    eval_args,
                    eval_accuracy_threshold,
                    serving_model_dir,
                    metadata_connection_config=None,
                    beam_pipeline_args=None,
                    ai_platform_training_args=None,
                    ai_platform_serving_args=None):

  """Implements the chicago taxi pipeline with TFX."""

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input=external_input(data_path))
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
  components.append(schema_gen)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(  
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  components.append(transform)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer_args = {
      'run_fn': run_fn,
      'transformed_examples': transform.outputs['transformed_examples'],
      'schema': schema_gen.outputs['schema'],
      'transform_graph': transform.outputs['transform_graph'],
      'train_args': train_args,
      'eval_args': eval_args,
      'custom_executor_spec':
          executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
  }
    
  if ai_platform_training_args is not None:
    trainer_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.GenericExecutor
            ),
        'custom_config': {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        }
    })
    
  trainer = Trainer(**trainer_args)
  components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
  components.append(model_resolver)

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='price')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='MeanSquaredError',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': eval_accuracy_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
  components.append(evaluator)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'model_blessing':
          evaluator.outputs['blessing'],
      'push_destination':
          pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)),
  }
  if ai_platform_serving_args is not None:
    pusher_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                           ),
        'custom_config': {
            ai_platform_pusher_executor.SERVING_ARGS_KEY:
                ai_platform_serving_args
        },
    })
  pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
  components.append(pusher)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )



