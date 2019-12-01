import os
from .logging import Logger
import traceback

L = Logger()
logger = L.get_logger()

DEFAULT_CONFIG_FILE = 'default.cfg'

# configuration for FedAvg
class Config():
    def __init__(self, config_file = 'default.cfg'):
        self.dataset = 'shakespeare'
        self.model = 'stacked_lstm'
        self.num_rounds = -1            # -1 for unlimited
        self.lr = 0.1
        self.eval_every = 3             # -1 for eval when quit
        self.clients_per_round = 10
        self.batch_size = 10
        self.seed = 0
        self.metrics_file = 'metrics'
        self.num_epochs = 1
        self.gpu_fraction = 0.2
        self.minibatch = None       # always None for FedAvg
        self.round_ddl = [150, 10]
        
        logger.info('read config from {}'.format(config_file))
        self.read_config(config_file)
        self.log_config()
        
        
    def read_config(self, filename = DEFAULT_CONFIG_FILE):
        if not os.path.exists(filename):
            logger.error('ERROR: config file {} does not exist!'.format(filename))
            assert False
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    param, value = line.split()
                    if param == 'num_rounds':
                        self.num_rounds = int(value)
                    elif param == 'learning_rate':
                        self.lr = float(value)
                    elif param == 'eval_every':
                        self.eval_every = int(value)
                    elif param == 'clients_per_round':
                        self.clients_per_round = int(value)
                    elif param == 'batch_size':
                        self.batch_size = int(value)
                    elif param == 'seed':
                        self.seed = int(value)
                    elif param == 'metrics_file':
                        self.metrics_file = str(value)
                    elif param == 'num_epochs':
                        self.num_epochs = int(value)
                    elif param == 'dataset':
                        self.dataset = str(value)
                    elif param == 'model':
                        self.model = str(value)
                    elif param == 'gpu_fraction':
                        self.gpu_fraction = float(value)
                    elif param == 'round_ddl':
                        mean, std = value.split(',')
                        self.round_ddl = [float(mean), float(std)]
                except Exception as e:
                    traceback.print_exc()
    
    def log_config(self):
        configs = vars(self)
        logger.info('================= Config =================')
        for key in configs.keys():
            logger.info('\t{} = {}'.format(key, configs[key]))
        logger.info('================= ====== =================')
        