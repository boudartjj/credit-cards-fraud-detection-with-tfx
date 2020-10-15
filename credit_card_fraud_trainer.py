from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
from tensorflow.keras import layers, models, optimizers
import os
import glob
import codecs
import numpy as np

_DENSE_FLOAT_FEATURE_KEYS = ['Amount', 'Time', 'V01', 'V02', 'V03']
_LABEL_KEY = 'Class'


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  train_dataset, train_labels = _input_fn(fn_args.train_files, tf_transform_output, 40)
  eval_dataset, eval_labels = _input_fn(fn_args.eval_files, tf_transform_output, 40)
        
  print(fn_args.serving_model_dir)
    
  #create model
  model = models.Sequential()
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(10, activation='relu'))
  model.add(layers.Dense(1, activation='relu'))
    
  model.compile(optimizer=optimizers.Adam(0.01), loss='MSE', metrics=['accuracy'])
    
  # train model
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
  model.fit(
    train_dataset,
    train_labels, 
    steps_per_epoch=fn_args.train_steps,
    validation_data=(eval_dataset, eval_labels), 
    validation_steps=fn_args.eval_steps,
    callbacks=[tensorboard_callback])
  model.fit(train_dataset, train_labels, epochs=10)
    
  #save model
  model.save("model.h5")

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')
    
def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  #print(transformed_feature_spec)

  files = glob.glob(file_pattern[0])
  dataset = _gzip_reader_fn(files).take(10000)

  dataset = dataset.map(lambda x: tf.io.parse_example(x, transformed_feature_spec))

  features = []
  labels = []
  for record in dataset:
    feature_row = []
    for feat in transformed_feature_spec:
        if feat in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS):
            feature_row.append(record[feat].numpy())
        if feat == _transformed_name(_LABEL_KEY):
            labels.append(record[feat].numpy())
    features.append(feature_row)

  return np.array(features), np.array(labels)

def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]

def _transformed_name(key):
  return key + '_xf'