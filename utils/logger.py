import logging

class Logger():
    def __init__(self, log_name='main'):
        self.log_name = log_name
        self.logger = None

    def get_logger(self):
        log_file = '{}.log'.format(self.log_name)
        logging.basicConfig(level = logging.INFO, filename=log_file, filemode='w', format = '%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('FL-type')
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(sh)
        self.logger.info('Log File: {}'.format(log_file))
        return self.logger

