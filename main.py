import os
import random
import pickle
import importlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from client import Client
from server import Server
from utils.data import read_data
from utils.logger import Logger
from utils.config import Config

random.seed(0)
np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tfe.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
MODEL_PARAMS = {'femnist.cnn': (0.05, 62)}


def init(dataset, model, alg):
    logger = Logger('out').get_logger()
    cfg = Config('default.cfg', logger)
    logger.info('Algorithm: %s ' % alg)

    model_path = '%s.%s' % (dataset, model)
    logger.info('Model: %s ' % model_path)
    ClientModel = getattr(importlib.import_module('model.'+model_path),'ClientModel')
    model = ClientModel(*MODEL_PARAMS[model_path])

    if os.path.exists('data/femnist/clients.pkl'):
        with open('data/femnist/clients.pkl', 'rb') as f:
            users, train_data, test_data = pickle.load(f)
    else:
        train_data_dir = os.path.join('data', dataset, 'data', 'train')
        test_data_dir = os.path.join('data', dataset, 'data', 'test')
        users, train_data, test_data = read_data(train_data_dir, test_data_dir)
        with open('data/femnist/clients.pkl', 'wb') as f:
            pickle.dump([users, dict(train_data), dict(test_data)], f)
    clients = [Client(model, logger, i+1, train_data[u], test_data[u], cfg) for i, u in enumerate(users)]
    logger.info('Total Client num: %d' % len(clients))
    clients = clients[:cfg.num_clients]
    logger.info('Clients in Used: {}'.format([c.id for c in clients]))
    server = Server(logger, model, clients, alg, cfg)

    return logger, cfg, server


# 一次 Round 为完成一轮训练，涉及到所有 client
# MA: 先训练 client 本地模型，返回参数，server 参数平均
#     client 本地模型参数: [num_epochs, batch_size]
# SSGD: 每个 client 用本地数据求梯度，返回梯度，server 梯度平均，更新参数
#       client 本地模型参数: [batch_size]
def sync_train(logger, cfg, server):
    now = 0
    for i in range(cfg.num_rounds):
        logger.info('========================= Round {} of {} =========================='.format(i+1, cfg.num_rounds))
        logger.info('------------------------ select deadline -------------------------')
        server.select_deadline(cfg.round_ddl)
        if server.alg == 'MA':
            logger.info('-------------------------- train params --------------------------')
            now += server.sync_get_params_or_grads()
            logger.info('Time: %.2f' % now)
            logger.info('-------------------------- update params -------------------------')
            server.update_params()
        elif server.alg == 'SSGD':
            logger.info('---------------------------- get grads ---------------------------')
            now += server.sync_get_params_or_grads()
            logger.info('Time: %.2f' % now)
            logger.info('---------------------------- apply grads -------------------------')
            server.apply_grads()
        if i % cfg.eval_every == 0:
            logger.info('------------------------------- test -----------------------------')
            logger.info('Train_acc: %.3f  Train_loss: %.3f' % server.test(set_to_use='train'))
            logger.info('Test_acc: %.3f  Test_loss: %.3f' % server.test(set_to_use='test'))

# 一次 Round 为某个 client 完成一次训练，且更新 server 参数
# ASGD、DC_ASGD: 每轮用 client 本地数据求本地参数梯度，更新 server 参数
#                client 本地模型参数: [batch_size]
def async_train(logger, cfg, server):
    for i in range(cfg.num_rounds):
        logger.info('========================= Round {} of {} ========================='.format(i+1, cfg.num_rounds))
        logger.info('---------------------------- get grads ---------------------------')
        now = server.async_get_grads()
        logger.info('Time: %.2f' % now)
        logger.info('---------------------------- apply grads -------------------------')
        server.apply_grads()
        if i % cfg.eval_every == 0:
            logger.info('------------------------------- test -----------------------------')
            logger.info('Train_acc: %.3f  Train_loss: %.3f' % server.test(set_to_use='train'))
            logger.info('Test_acc: %.3f  Test_loss: %.3f' % server.test(set_to_use='test'))


if __name__ == '__main__':
    # sync: ['MA', 'SSGD']
    # async: ['ASGD', 'DC_ASGD']
    alg = 'ASGD'
    logger, cfg, server = init('femnist', 'cnn', alg)

    if alg in ['MA', 'SSGD']:
        sync_train(logger, cfg, server)
    elif alg in ['ASGD', 'DC_ASGD']:
        async_train(logger, cfg, server)
