import queue
import numpy as np
import tensorflow as tf


class Server:
    def __init__(self, logger, model, clients, alg, cfg):
        self.logger = logger
        self.model = model
        self.params = model.get_params()
        self.alg = alg
        self.num_epochs = cfg.num_epochs
        self.batch_size = cfg.batch_size
        self.deadline = None
        self.update_frac = cfg.update_frac
        self.clients = clients
        self.clients_params = []
        self.clients_grads = []
        self.DC_lam = 0.05
        if alg in ['ASGD', 'DC_ASGD']:
            self.cq = queue.PriorityQueue()
            for c in clients:
                c.params = model.get_params()
                self.cq.put((c.get_total_time(self.alg, self.batch_size),c))
        
    def select_deadline(self, round_ddl):
        self.deadline = np.random.normal(round_ddl[0], round_ddl[1])
        self.logger.info('selected deadline: %.2f' % self.deadline)

    def sync_get_params_or_grads(self):
        simulate_time = 0
        for c in self.clients:
            c_total_time = c.get_total_time(self.alg, self.batch_size)
            if c_total_time > self.deadline:
                simulate_time = self.deadline
                self.logger.info('client %d, use time %.2f, failed: timeout!' % (c.id, c_total_time))
            else:
                simulate_time = max(simulate_time,c_total_time)
                if self.alg == 'MA':
                    c_params, nums = c.train_params(self.params, self.num_epochs, self.batch_size)
                    self.clients_params.append((c, c_params, nums))
                    self.logger.info('client %d, use time %.2f, upload params successfully!' % (c.id, c_total_time))
                    self.logger.info('{} of {} clients upload params successfully!'.format(len(self.clients_params), len(self.clients)))
                elif self.alg == 'SSGD':
                    c_grads, nums = c.cal_grads(self.batch_size, self.params)
                    self.clients_grads.append((c, c_grads, nums))
                    self.logger.info('client %d, use time %.2f, upload grads successfully!' % (c.id, c_total_time))
                    self.logger.info('{} of {} clients upload grads successfully!'.format(len(self.clients_grads), len(self.clients)))
        return simulate_time

    def async_get_grads(self):
        now, c = self.cq.get()
        c_grads, nums = c.cal_grads(self.batch_size)
        self.clients_grads.append((c, c_grads, nums))
        self.logger.info('client {} upload grads successfully!'.format(c.id))
        self.cq.put((now+c.get_total_time(self.alg, self.batch_size),c))
        return now
    
    def update_params(self):
        if len(self.clients_params)/len(self.clients) >= self.update_frac:
            total_weight = 0
            sum_params = [0]*len(self.clients_params[0][1])
            for (c, c_params, nums) in self.clients_params:
                total_weight += nums
                for i, v in enumerate(c_params):
                    sum_params[i] += (nums*v)
            self.params = [params/total_weight for params in sum_params]
            self.logger.info('update params succeed.')
        else:
            self.logger.info('update params failed.')
        self.clients_params = []
        
    def apply_grads(self):
        if self.alg in ['ASGD', 'DC_ASGD'] or len(self.clients_grads)/len(self.clients) >= self.update_frac:
            total_weight = 0
            sum_grads = [0]*len(self.clients_grads[0][1])
            for (c, c_grads, nums) in self.clients_grads:
                total_weight += nums
                for i, v in enumerate(c_grads):
                    sum_grads[i] += (nums*v)
            grads = [grads/total_weight for grads in sum_grads]
            if self.alg == 'DC_ASGD':
                c_params = self.clients_grads[0][0].params
                for i in range(len(grads)):
                    grads[i] = grads[i]+self.DC_lam*tf.multiply(tf.multiply(grads[i],grads[i]),self.params[i]-c_params[i])
            self.model.set_params(self.params)
            self.model.apply_grads(grads)
            self.params = self.model.get_params()
            if self.alg in ['ASGD', 'DC_ASGD']:
                for (c, c_grads, nums) in self.clients_grads:
                    c.params = self.model.get_params()
            self.logger.info('apply grads successfully.')
        else:
            self.logger.info('apply grads failed.')
        self.clients_grads = []

    def test(self, set_to_use):
        accs, losses, nums = [], [], []
        for c in self.clients:
            c_metrics = c.test(self.params, set_to_use)
            accs.append(c_metrics['acc'])
            losses.append(c_metrics['loss'])
            nums.append(c_metrics['num'])
        acc = np.average(accs, weights=nums)
        loss = np.average(losses, weights=nums)
        return acc, loss
