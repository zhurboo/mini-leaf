import random
import warnings
import timeout_decorator
import sys

from utils.logging import Logger

L = Logger()
logger = L.get_logger()

class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.deadline = 1 # < 0 for unlimited
        # TODO change upload time to upload spead (upload time = upload num / upload speed)
        self.upload_time = 10

    
    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        # TODO
        # change upload time to upload spead (upload time = upload num / upload speed) 
        
        train_time_limit = self.get_train_time_limit()
        logger.debug('train_time_limit: {}'.format(train_time_limit))
        
        @timeout_decorator.timeout(train_time_limit)
        def train_inner(self, num_epochs=1, batch_size=10, minibatch=None):
            if minibatch is None:
                data = self.train_data
                comp, update = self.model.train(data, num_epochs, batch_size)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}

                # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                num_epochs = 1
                comp, update = self.model.train(data, num_epochs, num_data)
            num_train_samples = len(data['y'])
            return comp, num_train_samples, update
        
        return train_inner(self, num_epochs, batch_size, minibatch)

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
    
    
    def set_deadline(self, deadline = -1):
        if deadline < 0:
            self.deadline = sys.maxsize
        else:
            self.deadline = deadline
        logger.debug('client {}\'s deadline is set to {}'.format(self.id, self.deadline))
    
    def set_upload_time(self, upload_time):
        if upload_time > 0:
            self.upload_time = upload_time
        else:
            logger.error('invalid upload time: {}'.format(upload_time))
            assert False
        logger.debug('client {}\'s upload_time is set to {}'.format(self.id, self.upload_time))
    
    def get_train_time_limit(self):
        if self.upload_time < self.deadline :
            return self.deadline - self.upload_time
        else:
            return 1