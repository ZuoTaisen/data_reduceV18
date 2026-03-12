# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:11:46 2023

@author: zuots
"""
import sys
from math import *
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
cpath = os.getcwd()
sys.path.append(cpath + '//' + r'../')
import data_dir
import time_func
DataFold = data_dir.DataFold



#directory = os.path.abspath(os.path.join(os.getcwd(), "../modules"))
#sys.path.append(directory)

class data_info():
    def __init__(self):
        self.DataFold = DataFold #r'/data/hanzehua/vsanstrans'
        self.L1 = 9920       #mm
        self.L2 = 12820       #mm
        self.A1 = 30          #mm
        self.A2 = 8           #mm
        self.SampleWidth = 1.3  #mm
        self.SampleHeight = 30 #mm
        self.SourceWidth = 2.55 #mm
        self.SourceHeight = 40 #mm
        self.DetectorWidth = 210  #mm
        self.DetectorHeight = 210  #mm
        self.DetectorPixelWidth = 0.82 #mm
        self.DetectorPixelHeight = 0.82 #mm
        self.const = 3956.2   
        self.ModToDetector = 34820  #mm
        self.D4ToMod = 34820 #mm
        self.TimeDelay = 52.8 #info.instrument_info.TimeDelayD3 #17.628#52.7     #18.628  #ms  6埃，2.2埃和4埃的延时分别为：52.7ms，19.32ms和35.135ms
        self.TOF = 40              #ms
        #self.TofBins = 100
        self.QMin = 0.0002        #A^-1
        self.QMax = 0.015 # 0.13 #0.05          #A^-1
        self.QBins = 60   
        self.loc = locals() 
        self.WaveBins = 250 #250 #5000
        self.TofBins = self.WaveBins
        self.XBins = 256
        self.YBins = 256
        self.ThetaMin = 0
        self.SampleThickness = 1   #mm
        self.ThetaMax = np.arctan(self.DetectorWidth/6/self.L2)
        self.DeltaLambdaRatio = 0.019*self.const/self.ModToDetector
        #self.XCenter = 65.9*self.TubeWidth - self.TubeWidth*64 + self.BankGap/2 # 65.9*self.TubeWidth - self.TubeWidth*64 + self.BankGap/2    # mm
        #self.YCenter = 124.37*self.TubeHeight - self.BankHeight/2 # 124.37*self.TubeHeight - self.BankHeight/2     # mm
        self.SampleThickness = 1   #mm
        self.QX = np.linspace(-1*self.QMax,self.QMax,self.QBins)
        #self.QX = np.linspace(self.QMin,self.QMax,self.QBins)
        self.QY = np.linspace(-1*self.QMax,self.QMax,self.QBins)
        self.StartWavelength = self.const*self.TimeDelay/self.ModToDetector
        self.StopWavelength = self.const*(self.TimeDelay+self.TOF)/self.ModToDetector
        self.WaveBand = self.StopWavelength - self.StartWavelength
        self.WaveBin = self.WaveBand/self.WaveBins
        self.WavelengthArray = np.arange(self.StartWavelength+self.WaveBin/2,self.StopWavelength,self.WaveBin)
        self.TOFArray = np.arange(self.StartWavelength*self.ModToDetector/self.const+self.WaveBin/2,self.StopWavelength*self.ModToDetector/self.const,self.WaveBin*self.ModToDetector/self.const)
        self.XArray = np.arange(-1*self.DetectorPixelWidth*self.XBins/2,self.DetectorPixelWidth*self.XBins/2,self.DetectorPixelWidth)
        self.YArray = np.arange(-1*self.DetectorPixelHeight*self.XBins/2,self.DetectorPixelHeight*self.XBins/2,self.DetectorPixelHeight)
        self.QArray = np.zeros(len(self.QX))[:,None]*np.zeros(len(self.QY))

    def save_file2(self,xx,yy,file_name):
        with open(file_name,'w') as f:
            for x,y in zip(xx,yy):
                print('{:<20.8f}{:>20.8f}'.format(x,y),file = f)
        f.close()

def get_q_min_max(RunNum):
    info = data_info()
    DataFold = info.DataFold
    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
    f = h5py.File(DataFileName, "r")    
    data1 = f["/csns/instrument/module4/histogram_data"][()] #got the left bank data
    #data2 = f["/csns/instrument/module4/histogram_data"][()] #got the right bank data
    data1 = np.array(np.sum(data1,axis = 0))
    #data2 = np.array(np.sum(data2,axis = 0))

    tof4 = np.array(f["/csns/instrument/module4/time_of_flight"][()])
    #tof32 = np.array(f["/csns/instrument/module4/time_of_flight"][()])
    #print(data1)
    #print(data2)
    #tof4 = tof4[data1>0]
    tof4 = tof4/1000 # ms
    #print(tof4)
    WavelengthArray4 = info.const*tof4/info.ModToDetector
    #print(WavelengthArray4)
    Qmax = 4*np.pi*np.sin(info.ThetaMax/2)/np.min(WavelengthArray4)
    #print(Qmax)
    return Qmax 

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

def time_info_diff(file_name):
    with open(file_name, 'r') as f:
        content = f.read()

    # 提取时间字符串
    start_time = re.search(r'"start_time_utc":\s*"([^"]+)"', content).group(1)
    end_time = re.search(r'"end_time_utc":\s*"([^"]+)"', content).group(1)

    # 计算时间差
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    time_diff = (end - start).total_seconds() / 60
    return start,end,time_diff

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

def load_data2(RunNum):
    datainfo = data_info()
    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
    infoFileName = DataFold + "/" + str(RunNum) + "/" + str(RunNum)
    f = h5py.File(DataFileName, "r")
    data1 = f["/csns/instrument/module4/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module4/histogram_data"][()] #got the right bank data
    startTime,endTime,useTime,useTimeMin = time_func.get_experimental_time_info(DataFold,RunNum)


    #startTime = f["/csns/start_time_utc"][()][0]
    #endTime = f["/csns/end_time_utc"][()][0]
    #useTime = time_diff(startTime,endTime)
    #useTimeMin = useTime[1]
    startPulse = get_start_pulse(infoFileName)
    endPulse = get_end_pulse(infoFileName)
    Pulses = endPulse - startPulse
    ProtonCharge = f["/csns/proton_charge"][()]
    try:
        freq_ratio = f["/csns/Freq_ratio"][()]
    except:
        freq_ratio = 1
    #print(freq_ratio)
    tmp = datainfo.WaveBins
    if int(RunNum[-7:]) >= 4102:
        TofPoints = 500*freq_ratio
    else:
        TofPoints = 5000
    tmp2 = int(TofPoints/tmp)
  
    tof3 = f["/csns/instrument/module4/time_of_flight"][()]
    print(data1.shape)
    print(data2.shape)
    print(tof3.shape)
    tofReshaped3 = np.average(np.reshape(tof3[:-1],(datainfo.TofBins,tmp2)),axis = 1)/1000 
    Data1Reshaped = np.reshape(data1,(256,256,tmp,tmp2))
    Data1Reshaped2 = np.sum(Data1Reshaped,axis = 3)
#    Data2Reshaped = np.reshape(data2,(256,256,tmp,tmp2))
#    Data2Reshaped2 = np.sum(Data2Reshaped,axis = 3) 
#    Data1Reshaped = np.reshape(data1,(64,250,5000))
#    Data2Reshaped = np.reshape(data2,(64,250,5000))

    DataStacked = Data1Reshaped2 #np.vstack((Data2Reshaped2,Data1Reshaped2))
    #DataStacked[0:80] = 0
    scale = 2.388E11
    
    print('The run number is : ' + str(RunNum))
    print('The run number is : ' + str(RunNum))
    print('Start time is:',startTime)
    print('End time is:',endTime)
    print('Used time is:',useTime,'seconds or ',np.round(useTimeMin,2),'Minutes')   
    print('The proton charge is: ' + str(ProtonCharge) + '\n')
    
    Counts1= np.sum(Data1Reshaped)
    print('Total counts of \nD4: ' + str(round(Counts1)))
    
    print('Count rate of  (n/s) \nD4: ' + str(round(Counts1/useTime)))

    
    NormedCounts1= np.sum(Data1Reshaped)/ProtonCharge*scale
    print('Proton Charge Normed counts of (n/PC*2.388E11) \nD4: ' + str(round(NormedCounts1)))

    TNormedCounts1= np.sum(Data1Reshaped)/ProtonCharge*scale/useTimeMin
    print('Time and Proton Charge Normed counts of (n/min/PC*2.388E11) \nD4: ' + str(round(TNormedCounts1)))

    pc_t = ProtonCharge/useTimeMin
    print('ProtonCharge/Time/1E9 (PC/Min/1E9):\n',pc_t/1E9)
    #DataStacked[:200,:,:] = 0 
    return DataStacked/ProtonCharge*scale,tofReshaped3

def save_matrix(matrix,FileName):
    with open(FileName,'w') as f:
        for i in matrix:
            for j in i:
                print('{:<20.8f}'.format(j),file = f, end = '')
            print('\n',file = f, end = '')
    f.close()

    
def plot_2d(DataStacked,save_name, show = True): 
    DataStacked2 = DataStacked.transpose(1,0,2) 
    datainfo = data_info()
    WavelengthArray = datainfo.const*tof3/datainfo.D4ToMod #-7 
    #print(WavelengthArray)
    falling = falling_distance(WavelengthArray,datainfo.L1,datainfo.L2)/1 #/4
    BankNames = get_variable_names(DataStacked)
    BankName = BankNames[0] #min(BankNames, key=len)
    counts = np.sum((DataStacked))
    D3XP = datainfo.XArray
    D3YP = datainfo.YArray
    DataStacked2 = falling_correct4(D3XP,D3YP,DataStacked2,falling)


    #BankNames = get_variable_names(DataStacked)
    #BankName = min(BankNames, key=len)
    Data2D = np.log10(np.sum(DataStacked2[:,:,:],axis = 2))
    #Data2D = np.log10(DataStacked)
    save_matrix(np.sum(DataStacked2,axis = 2),save_name + '_MatrixPlot_D3.dat')
    counts = np.sum((DataStacked))
    plt.figure(1)
    ax = plt.matshow(np.transpose(Data2D),cmap = plt.cm.jet)
    plt.title("2D map of " + save_name  + BankName +' counts:'+ str(counts))
    plt.colorbar()#.set_label(Data2D)
    plt.ylabel('Vertical dimension(Pixels)')
    plt.xlabel('Horizontal diemension(Pixels)')
    plt.savefig(save_name + '_MatrixPlot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)

    if show == True:
        plt.show()
    plt.close()

def plot_lambda(DataStacked,save_name, show = True):  
    datainfo = data_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    WavelengthArray = datainfo.const*tof3/datainfo.D4ToMod 
    WavePlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    #plt.plot(datainfo.WavelengthArray,WavePlot)
    with open(save_name + 'lambda.dat','w') as f:
        for i in range(len(WavePlot)):
            print(WavelengthArray[i],'  ',WavePlot[i],file = f)
    plt.plot(WavelengthArray, WavePlot)
    plt.xlabel('Neutron wavelegth (Angstrom)')
    plt.ylabel('Counts (n/A)')
    plt.title("Wavelength spectra of " + save_name  + BankName +' counts:'+ str(counts))
    plt.savefig(save_name+'_Lambda_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(datainfo.WavelengthArray,WavePlot,save_name +'_WavelengthPlot_D3.dat')
    if show == True:
        plt.show()
    plt.close()

def plot_tof(DataStacked,save_name, show = True):  
    datainfo = data_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    TOFPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    plt.plot(tof3,TOFPlot)
    plt.title("TOF of " + save_name  + BankName +' counts:'+ str(counts))
    plt.xlabel('Time-of-Flight (ms)')
    plt.ylabel('Counts (n/ms)')
    plt.savefig(save_name+'_TOF_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(tof3,TOFPlot,save_name +'_TOFPlot_D3.dat')    
    if show == True:
        plt.show()
    plt.close()

def falling_distance(wavelength,L_1,L_2):
    '''input a neutron wavelenth (A) and L1(mm), output the falling distance of the neutron (mm) '''
    B = 3.073E-9*100
    L = L_1 + L_2
    y = B*wavelength**2*L*(L_1-L)         #  公式来源： Bouleam SANS Tool Box: Chapter 17 - GRAVITY CORRECTRING PRISMS
    y = y/1000                            #mm
    return y    #mm

            
def falling_correct(X, Y, Counts, dY):
    """
    计算米粒在Y方向移动后的Y方向分布
    
    参数:
        X: X坐标点 (m个点的矩阵)
        Y: Y坐标点 (n个点的矩阵)
        Counts: 三维坐标点上的计数 (m×n×j的矩阵)
        dY: Y方向的移动距离 (j个点的矩阵)
    
    返回:
        Y方向的米粒分布 (n个点的矩阵)
    """
    m, n, j = Counts.shape
    
    # 检查输入维度是否匹配
    if len(X) != m:
        raise ValueError("X坐标点数与Counts的第一维不匹配")
    if len(Y) != n:
        raise ValueError("Y坐标点数与Counts的第二维不匹配")
    if len(dY) != j:
        raise ValueError("dY的长度与Counts的第三维不匹配")
    
    # 创建结果数组，初始化为0
    result = np.zeros(n)
    
    # 对于每个Z层（j）
    for z in range(j):
        # 获取当前Z层的移动距离
        current_dY = dY[z]
        
        # 计算移动后的Y坐标
        moved_Y = Y + current_dY
        
        # 找到移动后仍在原始Y范围内的索引
        valid_indices = np.where((moved_Y >= Y[0]) & (moved_Y <= Y[-1]))[0]
        
        # 对于每个有效的Y位置
        for orig_y_idx in valid_indices:
            # 计算移动后的Y位置
            new_y_pos = moved_Y[orig_y_idx]
            
            # 找到最近的Y坐标点
            new_y_idx = np.argmin(np.abs(Y - new_y_pos))
            
            # 将米粒数加到结果中
            result[new_y_idx] += np.sum(Counts[:, orig_y_idx, z])
    
    return result 

def falling_correct2(X, Y, Counts, dY):
    """
    计算米粒在Y方向移动后的Y方向分布
    
    参数:
        X: X坐标点 (m个点的矩阵)
        Y: Y坐标点 (n个点的矩阵)
        Counts: 三维坐标点上的计数 (m×n×j的矩阵)
        dY: Y方向的移动距离 (j个点的矩阵)
    
    返回:
        Y方向的米粒分布 (n个点的矩阵)
    """
    m, n, j = Counts.shape

    # 检查输入维度是否匹配
    if len(X) != m:
        raise ValueError("X坐标点数与Counts的第一维不匹配")
    if len(Y) != n:
        raise ValueError("Y坐标点数与Counts的第二维不匹配")
    if len(dY) != j:
        raise ValueError("dY的长度与Counts的第三维不匹配")

    # 创建结果数组，初始化为0
    result = np.zeros((m,n,j))

    # 对于每个Z层（j）
    for z in range(j):
        # 获取当前Z层的移动距离
        current_dY = dY[z]

        # 计算移动后的Y坐标
        moved_Y = Y + current_dY

        # 找到移动后仍在原始Y范围内的索引
        valid_indices = np.where((moved_Y >= Y[0]) & (moved_Y <= Y[-1]))[0]

        # 对于每个有效的Y位置
        for orig_y_idx in valid_indices:
            # 计算移动后的Y位置
            new_y_pos = moved_Y[orig_y_idx]

            # 找到最近的Y坐标点
            new_y_idx = np.argmin(np.abs(Y - new_y_pos))

            # 将米粒数加到结果中
            result[:,new_y_idx,z] += Counts[:,orig_y_idx,z]

    return result

def falling_correct3(X, Y, Counts, dY):
    """
    计算三维矩阵中米粒沿Y方向移动后的新矩阵。
    
    参数：
    X: 1D array, 形状 (m,), X坐标点
    Y: 1D array, 形状 (n,), Y坐标点
    Counts: 3D array, 形状 (m, n, j), 每个格点的米粒计数
    dY: 1D array, 形状 (j,), 每个Z层的Y方向移动距离
    
    返回：
    new_counts: 3D array, 形状 (m, n, j), 移动后的米粒计数矩阵
    """
    # 获取矩阵尺寸
    m = len(X)  # X维度
    n = len(Y)  # Y维度
    j = len(dY) # Z维度
    
    # 验证输入尺寸
    if Counts.shape != (m, n, j):
        raise ValueError("Counts矩阵的形状必须为 (m, n, j)")
    
    # 初始化新矩阵，形状与Counts相同
    new_counts = np.zeros((m, n, j), dtype=Counts.dtype)
    
    # 遍历每个格点 (x, y, z)
    for x in range(m):
        for y in range(n):
            for z in range(j):
                # 计算移动后的Y坐标
                y_new = y + dY[z]
                # 检查y_new是否在有效范围内
                if 0 <= y_new < n:
                    # 将米粒计数累加到新位置 (x, y_new, z)
                    new_counts[x, y_new, z] += Counts[x, y, z]
                # 如果y_new超出范围，米粒被移除，不计入新矩阵
    
    return new_counts

def falling_correct4(X, Y, Counts, dY):
    """
    计算Y方向移动后的三维米粒分布矩阵
    
    参数:
        X (np.ndarray): X坐标数组（m个元素）
        Y (np.ndarray): Y坐标数组（n个元素，必须单调递增）
        Counts (np.ndarray): 原始三维计数矩阵（m×n×j）
        dY (np.ndarray): Y方向移动距离数组（j个元素）
    
    返回:
        np.ndarray: 移动后的三维计数矩阵（m×n×j）
    """
    dY = -1*dY
    m, n, j = Counts.shape
    
    # 验证输入维度
    if len(X) != m or len(Y) != n or len(dY) != j:
        raise ValueError("输入维度不匹配")
    
    # 确保Y坐标是单调递增的
    if not np.all(np.diff(Y) > 0):
        raise ValueError("Y坐标必须是单调递增的")
    
    Y_min = Y[0]
    Y_max = Y[-1]
    new_counts = np.zeros_like(Counts)
    
    # 预计算Y坐标差值矩阵
    Y_expanded = Y[:, np.newaxis]  # 转换为列向量
    
    for z in range(j):
        # 当前层的移动距离
        delta = dY[z]
        
        # 计算移动后的Y坐标
        moved_Y = Y + delta
        
        # 确定有效移动范围
        valid_mask = (moved_Y >= Y_min) & (moved_Y <= Y_max)
        valid_y_orig = np.where(valid_mask)[0]
        
        if not valid_y_orig.size:
            continue
        
        # 计算新Y坐标对应的索引
        y_new_positions = moved_Y[valid_y_orig]
        
        # 向量化计算最近邻索引
        diffs = np.abs(Y_expanded - y_new_positions)
        y_new_indices = np.argmin(diffs, axis=0)
        
        # 使用向量化操作更新新矩阵
        new_counts[y_new_indices,:, z] += Counts[valid_y_orig,:, z]
    
    return new_counts

def plot_xy(DataStacked,save_name, show = True):  
    datainfo = data_info()
    WavelengthArray = datainfo.const*tof3/datainfo.D4ToMod #-7 
    #print(WavelengthArray)
    falling = falling_distance(WavelengthArray,datainfo.L1,datainfo.L2)/4
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    D3XP = datainfo.XArray
    D3YP = datainfo.YArray
    YPlot = falling_correct(D3XP,D3YP,DataStacked,falling)
    #D3YP2 = D3YP[:,None] - falling
    #YPlot = np.sum(np.sum(DataStacked,axis = 1),axis = 1)
    plt.figure(1)
    plt.plot(D3YP, YPlot)
    plt.title("Vertical axis of " + save_name  + BankName +' counts:'+ str(counts))
    plt.xlabel('Vertical dimension (mm)')
    plt.ylabel('Counts (n/mm)')
    plt.savefig(save_name+'_YPlot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(D3YP,YPlot,save_name +'_YData_D3'+ '.dat')

    plt.figure(2)    
    XPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 1)
    plt.plot(D3XP, XPlot)
    plt.title("Horizontal axis of " + save_name  + BankName +' counts:'+ str(counts))
    plt.xlabel('Horizontal dimension (mm)')
    plt.ylabel('Counts (n/mm)')
    plt.savefig(save_name+'_XPlot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(D3XP,XPlot,save_name +'_XData_D3'+ '.dat')
    if show == True:
        plt.show()
    plt.close()


def plot_2d_q(DataStacked,save_name, Qmin,Qmax, show = True):
    Info = data_info()
    counts = np.sum((DataStacked))

    D3XP = Info.XArray
    D3YP = Info.YArray

    D3ThetaX = np.arctan(D3XP/Info.L2)
    D3ThetaY = np.arctan(D3YP/Info.L2)
    
    WavelengthArray = Info.const*tof3/Info.D4ToMod 
    D3QX = 4*np.pi*np.sin(D3ThetaX[:,None]/2)/WavelengthArray
    D3QY = 4*np.pi*np.sin(D3ThetaY[:,None]/2)/WavelengthArray
    QX = np.linspace(Qmin,Qmax,Info.QBins)
    QY = np.linspace(Qmin,Qmax,Info.QBins)
    QArray = np.zeros(len(QX))[:,None]*np.zeros(len(QY))
#    QX = Info.QX
#    QY = Info.QY
#    QArray = Info.QArray
    QDict = {}
    
    for i in range(len(Data3D)):
        for j in range(len(Data3D[i])):
            for k in range(len(WavelengthArray)):
                Tmp1 = Info.QBins - len(QX[QX > D3QX[i,k]])-1
                Tmp2 = Info.QBins - len(QY[QY > D3QY[j,k]])-1
               # QArray[Tmp2,Tmp1] += Data3D[i,j,k]
                if (Tmp2,Tmp1) in QDict.keys():
                    QDict[(Tmp2,Tmp1)].append([i,j,k])
                else:
                    QDict[(Tmp2,Tmp1)] = []
                    QDict[(Tmp2,Tmp1)].append([i,j,k])
    
    
    #if os.path.exists(r'QDictFileD3' + str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.npy'):
    #np.save('QDictFileD3.npy',QDict)
    
    #QDictLoaded = np.load('QDictFileD3.npy').item()
    
    for key, items in QDict.items():
        #print(QDictLoaded)
        items = np.array(items)
        QArray[key[0],key[1]] = np.sum(Data3D[items[:,0],items[:,1],items[:,2]])
    
    
    QArray_2 = np.log10(QArray)
    plt.figure(1)
    plt.figure(figsize=(5, 5))
    ax = plt.matshow(QArray_2,cmap = plt.cm.jet)
    plt.xlabel('QX')
    plt.ylabel('QY')
    plt.savefig(RunNum + '_Q_Matrix_PlotD3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    np.save(RunNum + 'QX.txt',QX)
    np.save(RunNum + 'QY.txt',QY)
    np.save(RunNum + 'QArray', QArray)
    plt.figure(2)
    plt.figure(figsize=(5, 5))
    plt.contour(QX,QY,QArray_2,500,cmap = 'hot')
    plt.xlabel('QX (Angstrom$^{-1}$)')
    plt.ylabel('QY (Angstrom$^{-1}$)')
    plt.savefig(RunNum + '_Q_XY_Plot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    if show == True:
        plt.show()
    plt.close()

loc = locals()
def get_variable_name(variable):
    for k,v in loc.items():
        if loc[k] is variable:
            return k


def get_variable_names(variable):
    names = []
    for k,v in loc.items():
        if loc[k] is variable:
            names.append(k)
    return names




#DataFold = r'/data/hanzehua/vsanstrans'


RunNum = sys.argv[1]

if RunNum[:3] == "RUN":
    pass
else:
    RunNum = r"RUN" + str('0'*(7-len(RunNum))) + RunNum 

print(RunNum)

QMinMax = get_q_min_max(RunNum)
Qmin = -1*QMinMax #0.06       #A^-1
Qmax = QMinMax          #A^-1  2.2-6.7A 取0.15   6-10.5埃取0.05

Info = data_info()
Data3D,tof3 = load_data2(RunNum)
D3 = Data3D
show = True
if len(sys.argv) == 3:
    Cpara = sys.argv[2]
    if Cpara == '2D':
        plot_2d(Data3D,RunNum, show)
    
    elif Cpara == 'lambda':
        plot_lambda(Data3D,RunNum, show)
    
    elif Cpara == 'tof':
        plot_tof(Data3D,RunNum, show)
    
    elif Cpara == 'xy':
        plot_xy(Data3D,RunNum, show)
    
    elif Cpara == '2DQ':
        plot_2d_q(Data3D,RunNum, Qmin,Qmax, show)
        
    else:
        plot_2d(Data3D,RunNum, show)
    
        plot_lambda(Data3D,RunNum, show)
    
        plot_tof(Data3D,RunNum, show)
    
        plot_xy(Data3D,RunNum, show)
    
        plot_2d_q(Data3D,RunNum,Qmin,Qmax,  show) 
         
elif len(sys.argv) == 2:

    plot_2d(Data3D,RunNum, show)

    plot_lambda(Data3D,RunNum, show)

    plot_tof(Data3D,RunNum, show)

    plot_xy(Data3D,RunNum, show)

    plot_2d_q(Data3D,RunNum,Qmin,Qmax,  show)   











