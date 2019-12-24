import numpy as np
from utils.data import random_data, batch_data

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
    def __init__(self, model, logger, id, train_data, eval_data, cfg):
        self.logger = logger
        self.model = model
        self.id = id
        self.train_data = train_data
        self.eval_data = eval_data
        self.device = Device(np.random.randint(3), cfg)
        self.deadline = None
        self.params = None

    def get_train_time(self, num):
        speed = np.random.normal(self.device.speed_u, self.device.speed_sigma)
        return num/speed

    def get_upload_time(self):
        upload_time = np.random.normal(self.device.upload_time_u, self.device.upload_time_sigma)
        return upload_time
    
    def get_total_time(self, alg, batch_size):
        if alg == 'MA' or batch_size == -1:
            nums = len(self.train_data['y'])
        else:
            nums = min(batch_size, len(self.train_data['y']))
        train_time = self.get_train_time(nums)
        upload_time = self.get_upload_time()
        return train_time+upload_time
    
    def train_params(self, server_params, num_epochs, batch_size):
        self.model.set_params(server_params)
        for epoch in range(num_epochs):
            for x, y in batch_data(self.train_data, batch_size):
                self.model.train(x, y)
        return self.model.get_params(), num_epochs*len(self.train_data['y'])
    
    def cal_grads(self, batch_size, server_params=None):
        if batch_size == -1 or batch_size >= len(self.train_data['y']):
            x, y = self.train_data['x'], self.train_data['y']
        else:
            x, y = random_data(self.train_data, batch_size)
        if server_params:
            self.model.set_params(server_params)
        else:
            self.model.set_params(self.params)
        return self.model.cal_grads(x, y), len(y)

    def test(self, server_params, set_to_use):
        self.model.set_params(server_params)
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test':
            data = self.eval_data
        acc, loss = self.model.test(data['x'], data['y'])
        return {'acc':acc, 'loss':loss, 'num':len(data['y'])}
