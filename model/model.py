import tensorflow as tf
from copy import deepcopy
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, lr):
        self.model = self.create_model()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        pass
    
    @abstractmethod
    def cal_loss(self, logits, labels):
        pass
    
    @abstractmethod
    def cal_acc(self, logits, labels):
        pass
    
    def get_params(self):
        return deepcopy(self.model.variables)

    def set_params(self, params):
        for variable, value in zip(self.model.variables, params):
            variable.load(value)

    def cal_grads(self, x, y):
        x = self.process_x(x)
        y = self.process_y(y)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.cal_loss(logits, y)
        grads = tape.gradient(loss, self.model.variables)
        return grads
    
    def apply_grads(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.variables))
    
    def train(self, x, y):
        grads = self.cal_grads(x, y)
        self.apply_grads(grads)
    
    def test(self, x, y):
        x = self.process_x(x)
        y = self.process_y(y)
        logits = self.model(x, training=False)
        return self.cal_acc(logits, y), self.cal_loss(logits, y)
