import numpy as np
import tensorflow as tf
from model.model import BaseModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential

class ClientModel(BaseModel):
    def __init__(self, lr, lam, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr, lam)

    def create_model(self):
        inputs = Input(shape=(28, 28, 1), dtype='float64')
        x = Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, kernel_size=(5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(120, activation='relu')(x)
        x = Dropout(0.7)(x)
        x = Dense(84, activation='relu')(x)
        x = Dropout(0.7)(x)
        outputs = Dense(self.num_classes)(x)
        return Model(inputs=inputs, outputs=outputs)

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch).reshape((-1, 28, 28, 1))

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
