import sys
from math import *
import numpy
import scipy
import scipy.optimize
from matplotlib import pyplot as plt
import os
import numpy as np
import h5py
import re
import glob
import scipy
import scipy.optimize
import sys
from datetime import datetime


def get_start_pulse(file):
    f = open(file,'r')
    txt = str(f.readlines())       
    start_pulse_id_match = re.search(r'"startPulseId":\s*(\d+)', txt) 
    ProtonCharge = int(start_pulse_id_match.group(1))
    return ProtonCharge

def get_end_pulse(file):
    f = open(file,'r')
    txt = str(f.readlines())
    end_pulse_id_match = re.search(r'"endPulseId":\s*(\d+)', txt)
    ProtonCharge = int(end_pulse_id_match.group(1))
    return ProtonCharge


def time_diff(time1,time2):

    # 定义两个字节格式的时间
    #time1 = b'2024-12-16 01:47:29'
    #time2 = b'2024-12-16 08:25:44'
    
    # 将字节格式转换为字符串
    time1_str = time1.decode('utf-8')
    time2_str = time2.decode('utf-8')
    
    # 定义时间格式
    time_format = '%Y-%m-%d %H:%M:%S'
    
    # 将字符串时间转换为 datetime 对象
    time1_obj = datetime.strptime(time1_str, time_format)
    time2_obj = datetime.strptime(time2_str, time_format)
    
    # 计算时间差
    time_difference = time2_obj - time1_obj
    
    # 计算差值的秒数和分钟数
    difference_seconds = time_difference.total_seconds()
    difference_minutes = difference_seconds / 60
    
    # 输出结果
    #print(f"时间差（秒）：{difference_seconds}秒")
    #print(f"时间差（分钟）：{difference_minutes}分钟")
    return difference_seconds,difference_minutes

def load_data2(RunNum):

    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
    infoFileName = DataFold + "/" + str(RunNum) + "/" + str(RunNum)
    #try:
    for i in range(1):
        f = h5py.File(DataFileName, "r")
        startTime = f["/csns/start_time_utc"][()][0]
        endTime = f["/csns/end_time_utc"][()][0]
        useTime = time_diff(startTime,endTime)
        startPulse = get_start_pulse(infoFileName)
        endPulse = get_end_pulse(infoFileName)
        Pulses = endPulse - startPulse
        ProtonCharge = f["/csns/proton_charge"][()]
        #print(RunNum,':' , ProtonCharge,startTime,endTime,useTime[1])
        #print(RunNum,':' , ProtonCharge,round(useTime[1],2),round(ProtonCharge/useTime[1]/1E8,2))
        print(RunNum,':' , ProtonCharge,round(useTime[1],2),round(ProtonCharge/Pulses/1E6,2))
    #except:
    #    pass
DataFold = r'/data/hanzehua/vsanstrans'
#RunNum = []
for i in range(12060,12086):
    RunNum = 'RUN00' + str(i)
    load_data2(RunNum)
