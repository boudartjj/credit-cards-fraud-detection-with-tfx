import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
from tensorflow.keras import layers, models, optimizers

# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    print(fn_args.train_files)
    print(fn_args.transform_output)
    print(fn_args.serving_model_dir)
    print(fn_args.train_steps)
    print(fn_args.eval_steps)
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = load_dataset(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = load_dataset(fn_args.eval_files, tf_transform_output, 40)
    
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
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])
    
    #save model
    model.save("model.h5")
    
def load_dataset(file_pattern: str,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
            dataset to combine in a single batch

    Returns:
        A dataset that contains (features, indices) tuple where features is a
            dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name("Class"))

    return dataset

def _transformed_name(key):
    return key + '_xf'

def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]

def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP')