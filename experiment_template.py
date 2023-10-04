# Experiment metadata

## Experiment number
exp_num = ''

## Parent Experiment
parent_exp = ''

## Dataset ID
dataset_id = ''

## Notes about the experiment. Especially changes from parent experiment
notes = ''

## Model ID
model_id = ''

## Sprint ID
sprint_id = ''

# Imports
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from functools import partial
import numpy as np
import pickle
import os
import datetime
from model_template import model # import the model from its own module

# Load data
## Consider saving file names and labels together as a numpy array or pandas dataframe

file_names_path =r"" # path to the pickle object containing the list of all file names
with open(file_names_path, 'rb') as file:
    file_names = pickle.load(file)

data_dir = r"" # local folder containing training arrays

## Create list of full paths to the data
file_paths = [os.path.join(data_dir, file_name) for file_name in file_names if file_name.endswith('.npy')]

## load list of labels for all the data
labels_path = r"" # path to the pickle object containing the list of all labels
with open(labels_path, 'rb') as file:
    labels = pickle.load(file)


# Prepare the TensorFlow dataset. This will between experiments

## define functions to process the data
def load_numpy_file(file_path, label):
    array = np.load(file_path, allow_pickle = True)
    array = np.ndarray.astype(array, np.float32)
    return array, label

def parse_function(file_path, label):
    array, label = tf.numpy_function(load_numpy_file, [file_path, label], [tf.float32, tf.int32])
    return array, label

def fixup_shape_11(images, labels):
    images.set_shape([224, 224, 11])
    labels.set_shape([])
    return images, labels

## Create tf.Dataset from file paths and labels and shuffle it
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.shuffle(buffer_size = len(dataset), reshuffle_each_iteration = False)

## Apply the processing functions
dataset = dataset.map(parse_function)
dataset = dataset.map(fixup_shape_11)

## Split the data into training, validation, and test sets. This setup uses 70 - 10 - 20 split between training, validation, and test sets
train_size = round(len(dataset) * 0.7)
train_ds = dataset.take(train_size)
test_val_ds = dataset.skip(train_size)
val_ds = test_val_ds.take(len(test_val_ds)//3)
test_ds = test_val_ds.skip(len(test_val_ds)//3)

bs = 8
train_ds = train_ds.batch(bs).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(bs).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(bs).prefetch(tf.data.AUTOTUNE)

# Data augmentation

# Set up directories to store results

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

CHECKPOINT_DIR = r"" # File path for saving model checkpoints
callbacks = [
    ModelCheckpoint(CHECKPOINT_DIR, monitor='val_accuracy', save_weights_only = True, save_best_only= True, mode = 'max'),
    tensorboard_callback
]

# Train the model
epochs = 2
model_fit = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = callbacks)

# Run evaluation on the test data
metrics = model.evaluate(test_ds, return_dict = True)

accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1 = metrics['f1_score']

# Record results in csv


# Calculate performance metrics
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
precision.update_state(y_true, y_pred)
recall.update_state(y_true, y_pred)
f1_score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())

tf.summary.scalar('precision', precision.result())
tf.summary.scalar('recall', recall.result())
tf.summary.scalar('f1_score', f1_score)

# Record experiment metadata and performance metrics
