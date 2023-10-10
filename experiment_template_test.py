# Experiment metadata

## Experiment number
exp_num = 'template_test4'

## Parent Experiment
parent_exp = 'template_test2'

## Dataset ID
dataset_id = 'original_development_data'

## Notes about the experiment. Especially changes from parent experiment
notes = 'Changed ModelCheckpoint to save_best_only = False'

## Model ID
model_id = 'dummy'

## Sprint ID
sprint_id = '1'

# Imports
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import EarlyStopping
from functools import partial
import numpy as np
import pickle
import os
import datetime
from model_template import model # import the model from its own module

# Load data
## Consider saving file names and labels together as a numpy array or pandas dataframe

file_names_path =r"G:\.shortcut-targets-by-id\1F5TKMAk_9oNKo13HfwksVmLXmT-4Wy2n\WBS_Project\Training_data\model_development_file_names_numpy.pickle" # path to the pickle object containing the list of all file names
with open(file_names_path, 'rb') as file:
    file_names = pickle.load(file)

data_dir = r"G:\.shortcut-targets-by-id\1F5TKMAk_9oNKo13HfwksVmLXmT-4Wy2n\WBS_Project\Training_data\All_Training_Numpys" # local folder containing training arrays

## Create list of full paths to the data
file_paths = [os.path.join(data_dir, file_name) for file_name in file_names if file_name.endswith('.npy')]

## load list of labels for all the data
labels_path = r"G:\.shortcut-targets-by-id\1F5TKMAk_9oNKo13HfwksVmLXmT-4Wy2n\WBS_Project\Training_data\model_development_labels.pickle" # path to the pickle object containing the list of all labels
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

CHECKPOINT_DIR = r"P:\Eli\Mining_for_the_Future\porphyry-copper-deposit-prediction\checkpoints" # File path for saving model checkpoints
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(filepath = CHECKPOINT_DIR, monitor='val_loss', save_weights_only = True, save_best_only= False, mode = 'min'),
    tensorboard_callback
]

# Train the model
epochs = 2
model_fit = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = callbacks_list)

# Run evaluation on the test data
metrics = model.evaluate(test_ds, return_dict = True)

accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
try:
    f1 = 2 * (precision * recall)/(precision + recall)
except:
    f1 = 999

# Record results in csv
exp_results = pd.DataFrame({
    "Exp_num": exp_num,
    "Parent": parent_exp,
    "Notes": notes,
    "Sprint_ID": sprint_id,
    "Dataset": dataset_id,
    "Model": model_id,
    "Accuracy": accuracy,
    "Recall": recall,
    "Precision": precision,
    "F1": f1
},
index = [0])

results = pd.read_csv(r"P:\Eli\Mining_for_the_Future\porphyry-copper-deposit-prediction\experiment_results.csv")

results = pd.concat([results, exp_results], axis = 0)

results.to_csv(r"P:\Eli\Mining_for_the_Future\porphyry-copper-deposit-prediction\experiment_results.csv", index = False)


