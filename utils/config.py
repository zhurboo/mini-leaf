import os

class Config():
    def __init__(self, config_file, logger):
        self.logger = logger
        self.num_clients = 10
        self.num_rounds = -1
        self.num_epochs = 1
        self.batch_size = 10
        self.eval_every = 1
        self.round_ddl = [1000, 0]
        self.update_frac = 0.5
        self.big_upload_time = [5.0, 1.0]
        self.mid_upload_time = [10.0, 1.0]
        self.small_upload_time = [15.0, 1.0]
        self.big_speed = [150.0, 1.0]
        self.mid_speed = [100.0, 1.0]
        self.small_speed = [50.0, 1.0]
        self.read_config(config_file)
        self.log_config()

    def read_config(self, filename):
        self.logger.info('Config File: {}'.format(filename))
        with open(filename, 'r') as f:
            for line in f:
                line = line.split()
                if line[0] == 'num_clients':
                    self.num_clients = int(line[1])
                elif line[0] == 'num_rounds':
                    self.num_rounds = int(line[1])
                elif line[0] == 'num_epochs':
                    self.num_epochs = int(line[1])
                elif line[0] == 'batch_size':
                    self.batch_size = int(line[1])
                elif line[0] == 'dataset':
                    self.dataset = str(line[1])
                elif line[0] == 'model':
                    self.model = str(line[1])
                elif line[0] == 'round_ddl':
                    self.round_ddl = [float(line[1]), float(line[2])]
                elif line[0] == 'update_frac':
                    self.update_frac = float(line[1])
                elif line[0] == 'big_upload_time':
                    self.big_upload_time = [float(line[1]), float(line[2])]
                elif line[0] == 'mid_upload_time':
                    self.mid_upload_time = [float(line[1]), float(line[2])]
                elif line[0] == 'small_upload_time':
                    self.small_upload_time = [float(line[1]), float(line[2])]
                elif line[0] == 'big_speed':
                    self.big_speed = [float(line[1]), float(line[2])]
                elif line[0] == 'mid_speed':
                    self.mid_speed = [float(line[1]), float(line[2])]
                elif line[0] == 'small_speed':
                    self.small_speed = [float(line[1]), float(line[2])]

    def log_config(self):
        configs = vars(self)
        self.logger.info('================= Config =================')
        for key in configs.keys():
            self.logger.info('\t{} = {}'.format(key, configs[key]))
        self.logger.info('==========================================')

