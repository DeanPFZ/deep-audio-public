import numpy as np
from .model import Model
import keras as ks
import keras.layers as ksl
ks.backend.clear_session()

class EnvNet(Model):
    def __init__(self, input_length=24000, num_classes=50):
        super().__init__()
        self.model = self.__make_model(input_length, num_classes)

    def __make_model(self, input_length, num_classes):
        model = ks.Sequential()

        # Section 1, conv1
        model.add(
            ksl.Conv1D(
                input_shape=(input_length,1),
                filters=40,
                kernel_size=8,
                strides=1,
                padding="same",
            )
        )
        model.add(ksl.normalization.BatchNormalization())
        model.add(ksl.Activation('relu'))

        # Section 2, conv2
        model.add(
            ksl.Conv1D(
                filters=40,
                kernel_size=8,
                strides=1,
                padding="same",
            )
        )
        model.add(ksl.normalization.BatchNormalization())
        model.add(ksl.Activation('relu'))

        # Section 3, max_pooling_2d
        model.add(ksl.MaxPooling1D(pool_size=(160)))

        # Section 4, swapaxes
        model.add(ksl.Permute((2,1)))
        model.add(ksl.Reshape((model.output_shape[1], model.output_shape[2], 1)))

        # Section 5, conv3
        model.add(
            ksl.Convolution2D(
                filters=50,
                kernel_size=(8,13),
                strides=1,
                data_format="channels_last",
            )
        )
        model.add(ksl.BatchNormalization())
        model.add(ksl.Activation('relu'))

        # Section 6, max_pooling_2d
        model.add(
            ksl.MaxPooling2D(pool_size=(3,3))
        )

        # Section 7, conv4
        model.add(
            ksl.Convolution2D(
                filters=50,
                kernel_size=(1,5),
                strides=1,
            )
        )
        model.add(ksl.BatchNormalization())
        model.add(ksl.Activation('relu'))

        # Section 8, max_pooling_2d
        model.add(ksl.MaxPooling2D(pool_size=(1,3)))

        # Section 9, F.droput(F.relu(self.fc5(h)), train=self.train)
        model.add(ksl.Flatten())
        model.add(ksl.Dense(4096))
        model.add(ksl.Activation('relu'))
        model.add(ksl.Dropout(0.5))

        # Section 10, F.dropout(F.relu(self.fc6(h)), train=self.train)
        model.add(ksl.Dense(4096))
        model.add(ksl.Activation('relu'))
        model.add(ksl.Dropout(0.5))

        # Section 11, fc7(h)
        model.add(ksl.Dense(num_classes))

        return model

    def get_model(self):
        return self.model

    def print_summary(self):
        print(self.model.summary())