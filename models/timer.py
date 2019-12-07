import json
import time
from datetime import datetime


class Timer:
    def __init__(self, uid, google=True):
        self.fmt = '%Y-%m-%d %H:%M:%S'
        self.refer_time = '2018-03-06 00:00:00'
        self.refer_second = time.mktime(datetime.strptime(self.refer_time, self.fmt).timetuple())
        self.trace_start, self.trace_end = None, None
        self.uid = uid
        with open('../data/ready_{}.json'.format('strict' if google else 'loose'), 'r', encoding='utf-8') as f:
            self.ready_time = json.load(f)  # uid -> [ready_start, ready_end]
            self.ready_time = self.ready_time[uid]  # list([ready_start, ready_end])
        with open('../data/uid2behavior_tiny.json', 'r', encoding='utf-8') as f:
            self.process_message(json.load(f))

    def process_message(self, d):
        message = d[self.uid]['messages']
        state = ['battery_charged_off', 'battery_charged_on', 'battery_low', 'battery_okay',
                 'phone_off', 'phone_on', 'screen_off', 'screen_on', 'screen_unlock']
        for s in state:
            message = message.replace(s, "\t" + s + "\n")
        message = message.replace('\x00', '').strip().split("\n")
        for mes in message:
            try:
                t = mes.strip().split("\t")[0].strip()
                sec = time.mktime(datetime.strptime(t, self.fmt).timetuple()) - self.refer_second
                if not self.trace_start:
                    self.trace_start = sec
                self.trace_end = sec
            except:
                pass

    def ready(self, round_start, time_window):
        """
        if client is ready at time: round_start + time_window
        :param round_start: round start time (reference time)
        :param time_window: execute time
        :return: True if ready at round_start + time_window
        """
        now = int(round_start + time_window - self.trace_start) % (int(self.trace_end - self.trace_start)) + self.trace_start
        for item in self.ready_time:
            if item[0] <= now <= item[1]:
                return True
        return False

    def get_available_time(self, time_start, time_window):
        """
        get available time in [time_start, time_start + time_window]
        :param time_start: t
        :param time_window:  delta t
        :return: time
        """

        def overlay(S, E, t0, t1):
            # overlay of [S, E] and [t0, t1]
            res = 0
            if t0 <= S <= t1 <= E:
                res += t1 - S
            elif S <= t0 <= t1 <= E:
                res += t1 - t0
            elif S <= t0 <= E <= t1:
                res += E - t0
            elif t0 <= S <= E <= t1:
                res += E - S
            return res

        start = int(time_start - self.trace_start) % (int(self.trace_end - self.trace_start)) + self.trace_start
        end = start + time_window
        available_time = 0

        if end <= self.trace_end:
            for item in self.ready_time:
                available_time += overlay(start, end, item[0], item[1])
        else:
            trace_available = 0
            for item in self.ready_time:
                available_time += overlay(start, self.trace_end, item[0], item[1])
                end_ = int(end - self.trace_start) % (int(self.trace_end - self.trace_start)) + self.trace_start
                available_time += overlay(self.trace_start, end_, item[0], item[1])
                trace_available += item[1] - item[0]
            available_time += trace_available * (end - self.trace_end) // (self.trace_end - self.trace_start)
        
        return available_time
