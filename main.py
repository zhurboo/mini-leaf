import os
import time
import importlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from client import Client
from server import Server
from utils.data import read_data
from utils.logger import Logger
from utils.config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_PARAMS = {'femnist.cnn': (0.01, 0.05, 62),
                'sent140.stacked_lstm': (0.0003, 0.05, 25, 2, 100),
                'shakespeare.stacked_lstm': (0.0003, 0.05, 80, 80, 256)}


# 模型、客户端、服务器初始化
def init(dataset, model, alg):
    logger = Logger('out').get_logger()
    cfg = Config('default.cfg', logger)
    logger.info('Algorithm: %s ' % alg)

    # 模型初始化
    model_path = '%s.%s' % (dataset, model)
    logger.info('Model: %s ' % model_path)
    ClientModel = getattr(importlib.import_module('model.'+model_path),'ClientModel')
    model = ClientModel(*MODEL_PARAMS[model_path])

    # 客户端初始化
    train_data_dir = os.path.join('data', dataset, 'data', 'train')
    test_data_dir = os.path.join('data', dataset, 'data', 'test')
    users, train_data, test_data = read_data(train_data_dir, test_data_dir)
    clients = [Client(logger, u, train_data[u], test_data[u], cfg) for u in users]
    for i,client in enumerate(clients):
        client.id = i+1
    logger.info('Total Client num: %d' % len(clients))
    #clients = np.random.choice(clients, cfg.num_clients, replace=False)
    clients = clients[:cfg.num_clients]
    logger.info('Clients in Used: {}'.format([c.id for c in clients]))

    # 服务器初始化
    server = Server(logger, model, clients, alg)

    return logger, cfg, server


# MA(Model Average)
# 一次 Round 为完成一轮训练
def MA_train(logger, cfg, server):
    now = 0
    for i in range(cfg.num_rounds):
        logger.info('========================= Round {} of {} ========================='.format(i+1, cfg.num_rounds))
        logger.info('--------------------- select deadline ---------------------')
        server.MA_set_deadline(cfg.round_ddl)
        logger.info('-------------------------- train --------------------------')
        now += server.MA_train_model(cfg.num_epochs, cfg.batch_size)
        logger.info('Time: %.2f' % now)

        logger.info('-------------------------- update -------------------------')
        server.MA_update_model(cfg.update_frac)

        if i % cfg.eval_every == 0:
            logger.info('--------------------------- test --------------------------')
            acc, loss = server.test_model(set_to_use='train')
            logger.info('Train_acc: %.3f  Train_loss: %.3f' % (acc, loss))
            acc, loss = server.test_model(set_to_use='test')
            logger.info('Test_acc: %.3f  Test_loss: %.3f' % (acc, loss))

# ASGD(Asynchronous Stochastic Gradient Descent)
# DC_ASGD(Asynchronous Stochastic Gradient Descent with Delay Compensation)
# 一次 Round 为某个客户端完成一次训练
def ASGD_train(logger, cfg, server):
    for i in range(cfg.num_rounds):
        logger.info('========================= Round {} of {} ========================='.format(i+1, cfg.num_rounds))
        logger.info('-------------------------- train --------------------------')
        now = server.ASGD_train_model()
        logger.info('Time: %.2f' % now)

        logger.info('-------------------------- update -------------------------')
        server.ASGD_update_model()

        if i % cfg.eval_every == 0:
            logger.info('--------------------------- test --------------------------')
            acc, loss = server.test_model(set_to_use='train')
            logger.info('Train_acc: %.3f  Train_loss: %.3f' % (acc, loss))
            acc, loss = server.test_model(set_to_use='test')
            logger.info('Test_acc: %.3f  Test_loss: %.3f' % (acc, loss))


if __name__ == '__main__':
    tfe.enable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    alg = 'MA' # ['MA','ASGD','DC_ASGD']
    logger, cfg, server = init('femnist', 'cnn', alg)
    if alg == 'MA':
        MA_train(logger, cfg, server)
    else:
        ASGD_train(logger, cfg, server)
