import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import pandas as pd


def plot_charge(file):
    img_dir = "../../data/img"
    fmt = '%Y-%m-%d %H:%M:%S'
    reference_time = "2018-03-06 00:00:00"
    color = ['black', 'blue', 'green', 'yellow', 'red']
    state = ['battery_charged_off', 'battery_charged_on', 'battery_low', 'battery_okay', 'phone_off', 'phone_on', 'screen_off', 'screen_on', 'screen_unlock']
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    a = list(range(len(data)))
    random.shuffle(a)
    fig_num = 4
    fig_size = 5
    a = a[:fig_num * fig_size]
    for i in range(fig_num):
        plt.figure()
        for j in range(fig_size):
            start, end = None, None
            plot_j = False
            message = data[str(a[i * fig_num + j])]['messages']
            for s in state:
                message = message.replace(s, "\t" + s + "\n")
            message = message.strip().split("\n")
            for mes in message:
                t, s = mes.strip().split("\t")
                t = t.strip()
                s = s.strip()
                try:
                    if s == 'battery_charged_on' and not start:
                        start = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                    elif s == 'battery_charged_off' and start:
                        end = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(datetime.strptime(reference_time, fmt).timetuple())
                        plt.plot([start, end], [j + 1, j + 1], color[j])
                        start, end = None, None
                        plot_j = True
                except:
                    pass
            if not plot_j:
                plt.plot([0], [j + 1], color[j])
        plt.xlabel("relative time")
        plt.ylabel("client id")
        plt.title("charged time")
        plt.savefig(os.path.join(img_dir, "relativeTime_clientID_{}.png".format(i+1)), format="png")
    # plt.show()

def gen_json():
    data = pd.read_csv("../../data/user_behavior_tiny.csv", encoding="utf-8")
    data = data['extra']
    d = dict()
    for i in range(len(data)):
        d[i] = json.loads(data[i])
    with open("../../data/user_behavior_tiny.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)


if __name__ == '__main__':
    plot_charge("../../data/user_behavior_tiny.json")