import pandas as pd
import numpy as np
from scipy.fft import fft
import os

path_1 = './part_1'

signals_dict = {}

class Sensor:
    def __init__(self, path, filename):
        self.filename = filename
        self.epoch, self.step, self.sensor = filename[:-4].split('-')
        self.step = float(self.step)/1000
        self.time_signals = pd.read_csv(path +'/'+filename)
        self.freq_signals = None

    def time_signal(self):
        time_array = np.arange(0,len(self.time_signals))
        time_array = time_array*self.step
        self.time_signals['time'] = time_array
        pass

    def freq_signal(self):
        for c in ['x','y','z']:
            data = np.array(self.time_signals[c])
            fft_array = fft(data)
            freq_array = np.arange(0,len(self.time_signals))*(1/max(self.time_signals['time']))
            pass
        

for idx, filename in enumerate(os.listdir(path_1)):
    signals_dict[idx] = Sensor(path_1, filename)
    signals_dict[idx].time_signal()
    signals_dict[idx].freq_signal()

pass