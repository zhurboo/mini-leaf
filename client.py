import sys
import random
import timeout_decorator
import numpy as np

class Device():
    def __init__(self, device_type, cfg):
        self.device_type = device_type
        if device_type == 0:
            self.upload_time_u = cfg.small_upload_time[0]
            self.upload_time_sigma = cfg.small_upload_time[1]
            self.speed_u = cfg.small_speed[0]
            self.speed_sigma = cfg.small_speed[1]
        elif device_type == 1:
            self.upload_time_u = cfg.mid_upload_time[0]
            self.upload_time_sigma = cfg.mid_upload_time[1]
            self.speed_u = cfg.mid_speed[0]
            self.speed_sigma = cfg.mid_speed[1]
        elif device_type == 2:
            self.upload_time_u = cfg.big_upload_time[0]
            self.upload_time_sigma = cfg.big_upload_time[1]
            self.speed_u = cfg.big_speed[0]
            self.speed_sigma = cfg.big_speed[1]


class Client:
    def __init__(self, logger, id, train_data, eval_data, cfg):
        self.logger = logger
        self.id = id
        self.train_data = train_data
        self.eval_data = eval_data
        self.device = Device(random.randint(0, 2), cfg)
        self.deadline = None
        self.params = None

    def get_train_time(self):
        speed = np.random.normal(self.device.speed_u, self.device.speed_sigma)
        return len(self.train_data['y'])/speed

    def get_upload_time(self):
        upload_time = np.random.normal(self.device.upload_time_u, self.device.upload_time_sigma)
        return upload_time

    def MA_train(self, model, num_epochs, batch_size):
        train_time = self.get_train_time()
        upload_time = self.get_upload_time()
        if train_time+upload_time > self.deadline:
            raise timeout_decorator.timeout_decorator.TimeoutError('timeout')
        else:
            update = model.MA_train(self.train_data, num_epochs, batch_size)
            return train_time+upload_time, len(self.train_data['y']), update

    def ASGD_train(self, model, server_params, alg):
        model.set_params(self.params)
        self.params = model.ASGD_train(self.train_data, server_params, alg)

    def test(self, model, set_to_use):
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test':
            data = self.eval_data
        return model.test(data)
