import queue
import numpy as np


class Server:
    def __init__(self, logger, model, clients, alg):
        self.logger = logger
        self.model = model
        self.params = model.get_params()
        self.alg = alg
        self.clients = clients
        if alg == 'ASGD' or alg == 'DC_ASGD':
            self.cq = queue.PriorityQueue()
            for c in clients:
                c.params = model.get_params()
                self.cq.put((c.get_train_time()+c.get_upload_time(),c))
        self.deadline = None
        self.updates = []

    def MA_set_deadline(self, round_ddl):
        self.deadline = np.random.normal(round_ddl[0], round_ddl[1])
        for c in self.clients:
            c.deadline = self.deadline
        self.logger.info('selected deadline: %.2f' % self.deadline)

    def MA_train_model(self, num_epochs, batch_size):
        simulate_time = 0
        for c in self.clients:
            self.model.set_params(self.params)
            simulate_time_c, num_samples, update = c.MA_train(self.model, num_epochs, batch_size)
            if simulate_time_c > self.deadline:
                simulate_time = self.deadline
                self.logger.info('client %d, use time %.2f, failed: timeout!' % (c.id, simulate_time_c))
            else:
                simulate_time = max(simulate_time,simulate_time_c)
                self.updates.append((c.id, num_samples, update))
                self.logger.info('client %d, use time %.2f, upload successfully!' % (c.id, simulate_time_c))
        return simulate_time

    def MA_update_model(self, update_frac):
        self.logger.info('{} of {} clients upload successfully!'.format(len(self.updates), len(self.clients)))
        if len(self.updates) / len(self.clients) >= update_frac:
            self.logger.info('round succeed, updating global model...')
            used_client_ids = [cid for (cid, num_samples, params) in self.updates]
            total_weight = 0
            base = [0] * len(self.updates[0][2])
            for (cid, num_samples, params) in self.updates:
                total_weight += num_samples
                for i, v in enumerate(params):
                    base[i] += (num_samples*v)
            self.params = [v / total_weight for v in base]
        else:
            self.logger.info('round failed.')
        self.updates = []

    def ASGD_train_model(self):
        now, c = self.cq.get()
        c.ASGD_train(self.model, self.params, self.alg)
        self.logger.info('client {} upload successfully!'.format(c.id))
        self.cq.put((now+c.get_train_time()+c.get_upload_time(),c))
        return now

    def ASGD_update_model(self):
        self.params = self.model.get_params()
        self.logger.info('update successfully!')

    def test_model(self, set_to_use):
        self.model.set_params(self.params)
        accs = []
        losses = []
        nums = []
        for c in self.clients:
            c_metrics = c.test(self.model, set_to_use)
            accs.append(c_metrics['accuracy'])
            losses.append(c_metrics['loss'])
            nums.append(c_metrics['num'])
        acc = np.average(accs, weights=nums)
        loss = np.average(losses, weights=nums)
        return acc, loss
