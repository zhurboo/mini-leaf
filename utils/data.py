import os
import json
import numpy as np
from collections import defaultdict


def random_data(data, num=10):
    pos = np.random.choice(range(len(data['y'])), min(num, len(data['y'])), replace=False)
    x = [data['x'][i] for i in pos]
    y = [data['y'][i] for i in pos]
    return x, y


def batch_data(data, batch_size):
    data_x = data['x']
    data_y = data['y']
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda : None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])
    clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    return train_clients, train_data, test_data

