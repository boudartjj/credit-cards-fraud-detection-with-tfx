
import tensorflow as tf
import tensorflow_transform as tft

_DENSE_FLOAT_FEATURE_KEYS = ['Amount', 'Time',
                             'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09',
                             'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                             'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
_LABEL_KEY = 'Class'

def preprocessing_fn(inputs):
    outputs = {}

    for key in _DENSE_FLOAT_FEATURE_KEYS:
      outputs[_transformed_name(key)] = tft.scale_to_z_score(
          _fill_in_missing(inputs[key]))
    
    outputs[_transformed_name(_LABEL_KEY)] = _fill_in_missing(inputs[_LABEL_KEY])

    return outputs

def _transformed_name(key):
  return key + '_xf'

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
