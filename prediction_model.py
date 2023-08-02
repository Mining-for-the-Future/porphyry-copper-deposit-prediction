# Builds a convolutional neural network and loads weights from prior training (please see ModelTraining_final.ipynb for details).

import tensorflow as tf
import tensorflow_hub as hub
from functools import partial

# Create a class for creating pooling layers across two bands
class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size = 2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat([shape[:-1], [groups, self.pool_size]], axis = 0)
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis = -1)
    
# Create a default 2D convolution layer
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")

# Build a model using the EffNetV2-XL(21k) classification model
model_handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"

model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.Input(shape = (224, 224, 11)),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (5, 5), strides = (1, 1), activation = 'relu'),
    DefaultConv2D(filters = 64),
    DepthPool(),
    DefaultConv2D(filters = 16),
    DefaultConv2D(filters = 8),
    DepthPool(),
    DefaultConv2D(filters = 3),

    hub.KerasLayer(model_handle, trainable=True),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(1,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          activation = 'sigmoid')
])

model.build((None,)+(224, 224)+(3,))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

weights_path = "best_weights03.h5"

model.load_weights(weights_path)

