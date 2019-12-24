import numpy as np
import tensorflow as tf
from model.model import BaseModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential

class ClientModel(BaseModel):
    def __init__(self, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr)

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
    
    def cal_loss(self, logits, labels):
        return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
    
    def cal_acc(self, logits, labels):
        corr_pred = tf.equal(tf.argmax(logits, axis=1), labels)
        return tf.reduce_mean(tf.cast(corr_pred, dtype=tf.float32))