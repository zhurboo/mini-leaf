import tensorflow as tf
import tensorflow.contrib.eager as tfe
from copy import deepcopy
from abc import ABC, abstractmethod
from utils.data import batch_data

class BaseModel(ABC):
    def __init__(self, lr, lam):
        self.model = self.create_model()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.lam = lam

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        pass

    def get_params(self):
        return deepcopy(self.model.variables)

    def set_params(self, params):
        for variable, value in zip(self.model.variables, params):
            variable.load(value)

    def cal_loss(self, logits, labels):
        return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    def cal_acc(self, logits, labels):
        corr_pred = tf.equal(tf.argmax(logits, axis=1), labels)
        return tf.reduce_mean(tf.cast(corr_pred, dtype=tf.float32))

    def MA_train(self, data, num_epochs, batch_size):
        for epoch in range(num_epochs):
            for batched_x, batched_y in batch_data(data, batch_size):
                x = self.process_x(batched_x)
                y = self.process_y(batched_y)
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    loss = self.cal_loss(logits, y)
                grads = tape.gradient(loss, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables))
        return self.get_params()

    def ASGD_train(self, data, server_params, alg):
        x = self.process_x(data['x'])
        y = self.process_y(data['y'])
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.cal_loss(logits, y)
        grads = tape.gradient(loss, self.model.variables)
        if alg == 'ASGD':
            self.set_params(server_params)
            self.optimizer.apply_gradients(zip(grads, self.model.variables))
        elif alg == 'DC_ASGD':
            for i in range(len(grads)):
                grads[i] = grads[i]+self.lam*tf.multiply(tf.multiply(grads[i],grads[i]),server_params[i]-self.model.variables[i])
            self.set_params(server_params)
            self.optimizer.apply_gradients(zip(grads, self.model.variables))
        return self.get_params()

    def test(self, data):
        x = self.process_x(data['x'])
        y = self.process_y(data['y'])
        logits = self.model(x, training=False)
        loss = self.cal_loss(logits, y)
        acc = self.cal_acc(logits, y)
        return {'accuracy': acc, 'loss': loss, 'num': len(y)}
