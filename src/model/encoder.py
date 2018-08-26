
import tensorflow as tf
from tensorflow.contrib import keras


class Encoder:
    """
        Uses the pretrained model InceptionV3 in Keras as the Encoder to
        produce image encodings.
    """

    def __init__(self, model='InceptionV3', learning=False):
        self.model = model
        self.learning = learning

    def get_cnn_encoder_prepro(self):
        """
            Creates and returns the pretrained CNN and its preprocessing unit,
            based on init params. Currently supports only InceptionV3.
            Can be extended to include others as well.
        """
        if self.model == 'InceptionV3':
            keras.backend.set_learning_phase(False)
            model_io = keras.applications.InceptionV3(include_top=False)
            input_preprocessing = \
                keras.applications.inception_v3.preprocess_input
            encoder = keras.models.Model(
                model_io.input,
                keras.layers.GlobalAveragePooling2D()(model_io.output))
            return encoder, input_preprocessing
        else:
            raise Exception(
                'Configured model {} not supported.'.format(self.model))
