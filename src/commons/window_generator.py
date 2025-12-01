import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns,
               excluded_columns=None,
               use_tf: bool = True,
               batch_size: int = 32,
               shuffle: bool = True):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.use_tf = use_tf
    self.batch_size = batch_size
    self.shuffle = shuffle

    if input_width <= 0 or label_width <= 0 or shift <= 0:
      raise ValueError("input_width, label_width, and shift must be positive integers")

    if train_df is None and val_df is None and test_df is None:
      raise ValueError("At least one of train_df, val_df, or test_df must be provided")

    if label_columns is None:
      raise ValueError("label_columns must be provided")

    # Work out the label column indices.
    self.label_columns = label_columns
    self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

    if excluded_columns is None:
      excluded_columns = []

    excluded = excluded_columns + label_columns

    numerical_features = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
      
    self.feature_columns = [col for col in numerical_features if col not in excluded]
    self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, model=None, plot_col='current_power', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
        
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [15min]')

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    
    if self.use_tf:
      ds = tf.keras.utils.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=self.shuffle,
          batch_size=self.batch_size)

      ds = ds.map(self.split_window)
      return ds

    # Non-TensorFlow alternative: return a tuple of (inputs, labels) numpy arrays
    # Build sliding windows over data
    num_windows = data.shape[0] - self.total_window_size + 1
    if num_windows <= 0:
      return (np.empty((0, self.input_width, data.shape[1]), dtype=np.float32),
              np.empty((0, self.label_width, data.shape[1] if self.label_columns is None else len(self.label_columns))),)

    # Create a view of windows: (num_windows, total_window_size, features)
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(self.total_window_size, data.shape[1]))
    # sliding_window_view adds an extra axis; reshape to expected
    windows = windows.reshape(num_windows, self.total_window_size, data.shape[1])

    # Split into inputs and labels using same logic as split_window
    inputs = windows[:, self.input_slice, :]
    labels = windows[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = np.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Optional shuffling
    if self.shuffle:
      idx = np.random.permutation(inputs.shape[0])
      inputs = inputs[idx]
      labels = labels[idx]

    # If batch_size is set, yield batches; else return full arrays
    if self.batch_size and self.batch_size > 0:
      # Return an iterator of batches (inputs_batch, labels_batch)
      def batch_iterator():
        for start in range(0, inputs.shape[0], self.batch_size):
          end = start + self.batch_size
          yield inputs[start:end], labels[start:end]
      return batch_iterator()

    return inputs, labels
    
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
  
  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])