# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:48:05 2023

@author: zuotaisen
"""

import sys
import os
import re
import glob
import math
import pickle
from datetime import datetime
import importlib as imp
import inspect

import numpy as np
import h5py
import scipy
import scipy.optimize
from scipy import interpolate
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as P
from numba import jit

import data_reduce_D3
import efficiency_calc
from input_module import InputModule
from output_module import OutputModule
from calculation_module import CalculationModule
import smooth_2D
from gravity_correction import falling_correction
from gravity_correction import falling_distance
import claude_correction
import yuanbao_correction as yuanbao

pickle.DEFAULT_PROTOCOL = 4

imp.reload(data_reduce_D3)

# 全局变量定义
global data_reduce
global data_reduce0

global lower
global upper




class data_reduce():
    def __init__(self, DataFold, InstrumentInfo):
        # 基础配置
        self.DataFold = DataFold
        self.info = InstrumentInfo
        self.cpath = os.getcwd()
        self.const = 3956.2
        
        # 仪器信息
        self.MaskSwitch = self.info.MaskSwitch      
        self.GISANS_mode = self.info.GISANS_mode 
        self.WaveBinsSelected = self.info.WaveBinsSelectedD3
        self.StartWave = self.info.StartWave
        self.StopWave = self.info.StopWave
        
        # 几何参数
        self.L1 = self.info.L1 + self.info.SampleDisplace  # 72.5 mm
        self.L2 = self.info.D3_L2 * self.info.L2_factor - self.info.SampleDisplace  # mm
        self.L1Direct = self.info.L1Direct  # mm
        self.A1 = self.info.A1  # mm
        self.A2 = self.info.A2  # mm
        self.A2Small = self.info.A2Small  # mm
        self.SamplePos = self.info.SamplePos + self.info.SampleDisplace  # 22000 mm
        self.ModToDetector = self.SamplePos + self.L2  # 33500 mm
        
        # 探测器参数
        self.DetFactor = self.info.DetFactor  # 5
        self.TubeWidth = 8.5  # mm
        self.TubeHeight = 4 * self.DetFactor  # mm
        self.BankGap = 6.4  # 12 #10 #10 #10 #12 #6 mm
        self.ModuleGap = 2  # 2
        self.ModuleWidth = self.TubeWidth * 16
        self.BankWidth = self.TubeWidth * 64 * 2 + self.ModuleGap * 6 + self.BankGap  # + self.TubeWidth
        self.BankHeight = 1000 * self.DetFactor - self.TubeHeight  # mm
        
        
        # 数据参数
        self.WaveMin = self.info.WaveMin  # 2.2 A
        self.WaveMax = self.info.WaveMax  # A
        self.TimeDelay = self.info.TimeDelayD3  # 49.85 #17.68#   49.85    #18.628  #ms  6埃，2.2埃和4埃的延时分别为：52.7ms，19.32ms和35.135ms
        self.TOF = self.info.TOF  # 40 ms
        self.QMin = self.info.QMin  # 0.001 A^-1
        self.QMax = self.info.QMax  # 2 A^-1
        self.QBins = self.info.QBins  # 120  
        self.QMax2D = self.info.QMax2D # 0.15
        self.QBins2D = int(self.QMax2D/0.12*250) 
        self.WaveBins = 250  # 5000
        
        # 其他参数
        self.CellScatteringFactor = self.info.CellScatteringFactor
        self.SampleScatteringFactor = self.info.SampleScatteringFactor
        self.I0Scale = self.info.IDirectBeamScale
        self.formated_time = self.get_now()
        self.XBins = 128
        self.YBins = 250  
        self.RBins = 200
        self.RMin = 1E-3  # 设置为一个很小的正数，避免除以零
        self.R =900      #mm
        self.DeltaLambdaRatio = 0.019*self.const/self.ModToDetector
        self.XCenter0 = self.info.XCenter  #lf.TubeWidth - self.BankWidth/2 + self.BankGap/2    #self.BankWidth/2    # mm
        self.YCenter0 = self.info.YCenter #elf.TubeHeight - self.BankHeight/2     # mm
        self.SampleThickness = 1   #mm
        #self.QX = np.logspace(log10(self.QMin),log10(self.QMax),self.QBins)
        #self.QX = np.linspace(self.QMin,self.QMax,self.QBins)
        self.QX = self.info.QX
        self.QY = np.zeros(len(self.QX))
        self.QX2D = np.linspace(-1*self.QMax2D,self.QMax2D,self.QBins2D)
        self.QY2D = np.linspace(-1*self.QMax2D,self.QMax2D,self.QBins2D)
 
        self.RArrayEdges = np.logspace(np.log10(self.RMin), np.log10(self.R), self.RBins + 1)
        self.RArray = np.sqrt(self.RArrayEdges[:-1] * self.RArrayEdges[1:])  # bin centers (geometric mean)
        self.L2_Array = np.sqrt(self.L2**2 + self.RArray**2)
        self.L_Array = self.SamplePos + self.L2_Array
        self.L = self.SamplePos + self.L2
        self.BeamStopStart = np.digitize(self.info.BeamStopDia/2,self.RArrayEdges)
        
        self.ThetaArray = np.arctan(self.RArray/self.L2_Array)        
        self.TofArray = np.linspace(self.TimeDelay,self.TimeDelay + self.TOF,self.WaveBins) + self.TOF/self.WaveBins/2

        # 使用与R相关的TofMin和TofMax，不同R值对应不同的飞行时间范围
        self.TofMin = self.L_Array[:,None]*self.StartWave/self.const # 单位：毫秒
        self.TofMax = self.L_Array[:,None]*self.StopWave/self.const   # 单位：毫秒
        
        # 确保WavelengthArrayBool是RBins×WaveBins的二维数组
        self.WavelengthArrayBool = (self.TofArray[None, :] > self.TofMin) * (self.TofArray[None, :] < self.TofMax)


        self.WavelengthArray2 = self.const*self.TofArray/self.L_Array[:,None]
        self.StartTof = (self.SamplePos + self.L2)*self.StartWave/self.const
        self.StopTof = (self.SamplePos+self.L2)*self.StopWave/self.const

        # 注释掉错误的WavelengthArrayBool定义，保留上面基于每个探测器位置计算的正确定义
        #self.WavelengthArrayBool = (self.WavelengthArray2 > self.StartTof)*(self.WavelengthArray2 < self.StopTof) 

        #self.StartWavelength = self.const*self.TimeDelay/self.ModToDetector
        #self.StopWavelength = self.const*(self.TimeDelay+self.TOF)/self.ModToDetector
        #self.WaveBand = self.StopWavelength - self.StartWavelength
        #self.WaveBin = self.WaveBand/self.WaveBins
#       self.WavelengthArray = np.arange(self.StartWavelength+self.WaveBin/2,self.StopWavelength,self.WaveBin)
        self.WavelengthArray = self.const*self.TofArray/self.L
        self.ModuleArray = np.arange(0,self.TubeWidth*16,self.TubeWidth)
        
        self.FirstArray = self.ModuleArray-self.BankWidth/2+self.TubeWidth/2
        self.ArrayDistance = self.ModuleWidth + self.ModuleGap
        self.XArrayLeft = np.concatenate((self.FirstArray,self.FirstArray+self.ArrayDistance,self.FirstArray+self.ArrayDistance*2,self.FirstArray+self.ArrayDistance*3)) 
        self.XArrayRight = np.concatenate((self.FirstArray+self.ArrayDistance*4+self.BankGap,self.FirstArray+self.ArrayDistance*5+self.BankGap,self.FirstArray+self.ArrayDistance*6+self.BankGap,self.FirstArray+self.ArrayDistance*7+self.BankGap)) - 2
        #self.XArrayLeft = np.arange(-1*self.TubeWidth*self.XBins/2, self.TubeWidth, self.TubeWidth) - self.BankGap/2
        #self.XArrayRight = np.arange(self.TubeWidth, self.TubeWidth*self.XBins/2, self.TubeWidth) + self.BankGap/2
        self.XArray0 = np.concatenate((self.XArrayLeft,self.XArrayRight))
        self.YArray = np.arange(-1*self.TubeHeight*self.YBins/2+self.TubeHeight/2,self.TubeHeight*self.YBins/2+self.TubeHeight/2,self.TubeHeight)# + self.YCenter
        self.XCenter = self.XArray0[int(self.XCenter0)]*(1-self.XCenter0%1) + self.XArray0[int(self.XCenter0)+1]*(self.XCenter0%1)#+self.TubeWidth
        self.YCenter = self.YArray[int(self.YCenter0)]*(1-self.YCenter0%1) + self.YArray[int(self.YCenter0)+1]*(self.YCenter0%1)
        #self.XArray0 = self.XArray
        self.XArray = self.XArray0 - self.XCenter
        with open('XArray.dat','w') as f:
            for i in range(len(self.XArray)):
                print(self.XArray[i],file = f)
        self.XArrayLeft = self.XArray_move_left()
        self.XArrayRight = self.XArray_move_right()
        self.XArray_div = self.divide_array(self.XArray)
        #self.YArray0 = self.YArray
        self.YArray = self.YArray + self.YCenter
        with open('YArray.dat','w') as f:
            for i in range(len(self.YArray)):
                print(self.YArray[i],file = f)
        
        self.FallingDistanceDirect = self.falling_distance(self.WavelengthArray,self.L1Direct,self.L2)
        self.FallingDistance = self.falling_distance(self.WavelengthArray,self.L1,self.L2)
        self.D3ThetaX = np.arctan(self.XArray/self.L2)
        self.D3ThetaXLeft = np.arctan(self.XArrayLeft/self.L2)
        self.D3ThetaXRight = np.arctan(self.XArrayRight/self.L2)
        self.D3ThetaX_div = np.arctan(self.XArray_div/self.L2)
        self.D3ThetaY = np.arctan((self.YArray[:,None]-self.FallingDistance)/self.L2)
        self.D3QX = 4*np.pi*np.sin(self.D3ThetaX[:,None]/2)/self.WavelengthArray
        self.D3QXLeft = 4*np.pi*np.sin(self.D3ThetaXLeft[:,None]/2)/self.WavelengthArray
        self.D3QXRight = 4*np.pi*np.sin(self.D3ThetaXRight[:,None]/2)/self.WavelengthArray
        self.D3QY = 4*np.pi*np.sin(self.D3ThetaY/2)/self.WavelengthArray
        self.QArray2d = np.zeros(len(self.QX2D))[:,None]*np.zeros(len(self.QY2D))
        self.CosAlpha = self.L2/np.sqrt(self.YArray**2 + self.L2**2)

    def divide_array(self,arr):
        div = (arr[2]-arr[1])/3
        matrix1 = arr - div
        matrix2 = arr
        matrix3 = arr + div
        interlaced_matrix = []
        for a, b, c in zip(matrix1, matrix2, matrix3):
            interlaced_matrix.extend([a, b, c])
        return np.array(interlaced_matrix)

    def expand_array(self,matrix):
        rows, cols = matrix.shape
        new_matrix = np.zeros((rows, cols * 3))
        for i in range(cols):
            value = matrix[:, i] / 3
            new_matrix[:, 3*i] = value   # 左侧列
            new_matrix[:, 3*i + 1] = value  # 中间列
            new_matrix[:, 3*i + 2] = value  # 右侧列
        return new_matrix

    def get_proton_charge(self,file):
        f = open(file,'r')
        txt = f.readlines()        
        pattern = re.compile(r'(\d+\.?\d*)</proton_charge>?')
        tmp = pattern.findall(str(txt))
        ProtonCharge = float(tmp[0])
        return ProtonCharge

    def XArray_move_left(self):
        XArrayLeft = np.zeros_like(self.XArray)
        XArrayLeft[1:] = self.XArray[1:] - (self.XArray[1:] - self.XArray[:-1])/3
        XArrayLeft[0] = self.XArray[0] - (self.XArray[1] - self.XArray[0])/3
        return XArrayLeft

    def XArray_move_right(self):
        XArrayRight = np.zeros_like(self.XArray)
        XArrayRight[:-1] = self.XArray[:-1] + (self.XArray[1:] - self.XArray[:-1])/3
        XArrayRight[-1] = self.XArray[-1] + (self.XArray[-1] - self.XArray[-2])/3
        return XArrayRight

    def load_mask(self):
        return InputModule.load_mask('D3')

    def get_proton_charge(self, file):
        return InputModule.get_proton_charge(file)

    def get_now(self):
        return InputModule.get_now()


    def falling_distance(self,wavelength,L_1,L_2):
        '''input a neutron wavelenth (A) and L1(mm), output the falling distance of the neutron (mm) '''
        B = 3.073E-9*100
        L = L_1 + L_2
        y = B*wavelength**2*L*(L_1-L)         #  公式来源： Bouleam SANS Tool Box: Chapter 17 - GRAVITY CORRECTRING PRISMS
        y = y/1000                            #mm
        return y    #mm

    def detector_group00(self,Height,Width,R,XBins,YBins,RBins,XCenter,YCenter):
        return CalculationModule.detector_group_d3(self, Height, Width, R, XBins, YBins, RBins, XCenter, YCenter)

    def detector_group(self,Height,Width,R,XBins,YBins,RBins,XCenter,YCenter):
        r = self.RArray #np.linspace(0,R,RBins)
        #r = np.logspace(log10(0.1),log10(R),RBins)
        FallingDistance = self.FallingDistance #self.falling_distance(self.WavelengthArray,self.L1,self.L2)
        GroupY = []
        GroupX = []
        for i in range(RBins):
            GroupY.append([])
            GroupX.append([])
        #    for k in range(self.WaveBins):
        #        GroupY[i].append([])
        #        GroupX[i].append([])
#        GroupedY = list(np.zeros(self.RBins)[:,None]*np.zeros(self.WaveBins))
#        GroupedX = list(np.zeros(self.RBins)[:,None]*np.zeros(self.WaveBins))
       # GroupedY = list(np.zeros(self.WaveBins)[:,None]*np.zeros(self.RBins))
       # GroupedX = list(np.zeros(self.WaveBins)[:,None]*np.zeros(self.RBins))
        #print(XCenter,YCenter,'This is the center')
        #print(FallingDistance)

        yy = self.YArray #np.linspace(-1*Height/2,Height/2,YBins)
        xx = self.XArray #np.linspace(-1*Width/2,Width/2,XBins)
        for i in np.arange(YBins):
            for j in np.arange(XBins):
                #Rij = np.sqrt((yy[i]+YCenter)**2 + (xx[j]-XCenter)**2)
                Rij = np.sqrt((yy[i])**2 + (xx[j])**2)
                # 使用 np.digitize 找到 Rij 对应的 bin 索引
                tmp = np.digitize(Rij, self.RArrayEdges) - 1
                # 边界情况处理
                if tmp < 0:
                    tmp = 0
                if tmp >= RBins:
                    tmp = RBins - 1
                GroupY[tmp].append(i)
                GroupX[tmp].append(j)  
        with open('./npyfiles/D3GroupX'+ str(round(self.XCenter,2)) + '-' + str(round(self.YCenter,2)) + '_' + str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.pkl', 'wb') as f:
            pickle.dump(GroupX, f)
        with open('./npyfiles/D3GroupY'+ str(round(self.XCenter,2)) + '-' +str(round(self.YCenter,2)) + '_' + str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.pkl', 'wb') as f:
            pickle.dump(GroupY, f)
#        np.save(r'./npyfiles/D3GroupX' + str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.pkl','wb',GroupX)
#        np.save(r'./npyfiles/D3GroupY' + str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.pkl','wb',GroupY)
        return GroupX,GroupY

    def solid_angle(self,Height,Width,R,XBins,YBins,RBins,XCenter,YCenter):
        return CalculationModule.solid_angle_d3(self, Height, Width, R, XBins, YBins, RBins, XCenter, YCenter)

    def solid_angle_2d(self,Height,Width,R,XBins,YBins,XCenter,YCenter):
        return CalculationModule.solid_angle_2d_d3(self, Height, Width, R, XBins, YBins, XCenter, YCenter)


    def grouping(self,DataReshaped,GroupX,GroupY,RBins):
        return CalculationModule.grouping_d3(self, DataReshaped, GroupX, GroupY, RBins)

    def grouping_mask(self,D3Data,XMinD3,XMaxD3,YMinD3,YMaxD3):       
        return CalculationModule.grouping_mask_d3(self, D3Data, XMinD3, XMaxD3, YMinD3, YMaxD3)

    def azimuthal_mask(self,D3Data,PhiMin,PhiMax):
        return CalculationModule.azimuthal_mask_d3(self, D3Data, PhiMin, PhiMax)

    def efficiency_matrix(self):
        calc1 = efficiency_calc.efficiency(self.L2,self.XArray,self.YArray,self.WavelengthArray)
        effic1 = calc1.pixel_efficiency()
        np.save(r'./npyfiles/D3_effic1_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',effic1)
        return effic1

    def time_diff(self,time1,time2):

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


    def get_experimental_time(self,RunNum):
        DataFileName = self.DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
        #RunInfoFileName = glob.glob(self.RunInfoFold + '/' + str(RunNum) + '/' + '**.xml')[0]
        f = h5py.File(DataFileName, "r")
        startTime = f["/csns/start_time_utc"][()][0]
        endTime = f["/csns/end_time_utc"][()][0]
        useTime = self.time_diff(startTime,endTime)
        useTimeMin = useTime[1]
        OutPut = [startTime,endTime,useTime,useTimeMin]
        return OutPut

    def get_experimental_time_info(self,RunNum):
        # 格式化RunNum为'RUN000XXX'形式
        if isinstance(RunNum, str) and RunNum.startswith('RUN'):
            # 如果RunNum已经包含RUN前缀，直接使用
            run_dir = RunNum
        else:
            # 否则添加RUN前缀并格式化
            run_dir = f"RUN{int(RunNum):07d}"
        FileName = os.path.join(self.DataFold, run_dir, run_dir)
        with open(FileName, 'r') as f:
            content = f.read()

        # 提取时间字符串
        start_time = re.search(r'"start_time_utc":\s*"([^"]+)"', content).group(1)
        end_time = re.search(r'"end_time_utc":\s*"([^"]+)"', content).group(1)

        # 计算时间差
        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        time_diff_min = (end - start).total_seconds() / 60
        time_diff_sec = (end - start).total_seconds()
        return [start,end,time_diff_sec,time_diff_min]


        
    def load_data(self, RunNum):
        Height = self.BankHeight
        Width = self.BankWidth
        R = self.R
        RBins = self.RBins
        XBins = self.XBins
        YBins = self.YBins
        XCenter = self.XCenter
        YCenter = self.YCenter
        
        DataStacked, ProtonCharge, GroupX, GroupY = self.load_data_origin(RunNum)
        DataYIntegrated = self.grouping(DataStacked, GroupX, GroupY, RBins)
        BeforeGrouping = np.sum(DataStacked)
        #print(np.sum(DataStacked)) 
        DataYIntegrated = DataYIntegrated * self.WavelengthArrayBool
        
        DataNormed = DataYIntegrated  # /ProtonCharge*9000*390701/10       
        DataNormed = DataNormed * self.WavelengthArrayBool
        #print(np.sum(DataNormed)) 
        AfterGrouping = np.sum(DataNormed)
        return DataNormed*BeforeGrouping/AfterGrouping, ProtonCharge

    def load_data_origin(self,RunNum):
        Height = self.BankHeight
        Width = self.BankWidth
        R = self.R
        RBins = self.RBins
        XBins = self.XBins
        YBins = self.YBins
        XCenter = self.XCenter
        YCenter = self.YCenter
        DataYIntegrated = []
        DataFileName = self.DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
        #ExpInfoFileName = self.DataFold + "/" + str(RunNum) + "/" + str(RunNum)         
        #RunInfoFileName = glob.glob(self.RunInfoFold + '/' + str(RunNum) + '/' + '**.xml')[0]  
        
        f = h5py.File(DataFileName, "r")
        data1 = f["/csns/instrument/module32/histogram_data"][()] #got the left bank data
        data2 = f["/csns/instrument/module31/histogram_data"][()] #got the right bank data
        ProtonCharge = f["/csns/proton_charge"][()]#ProtonCharge = np.sum(f["/csns/logs/proton_charge/value"][()])
        #DataReshaped = np.reshape(data,(64,250,5000))
        tmp = self.WaveBins
        try:
            freq_ratio = f["/csns/Freq_ratio"][()]
        except:
            freq_ratio = 1
        if int(RunNum[-6:]) >= 4102:
            TofPoints = 500*freq_ratio
        else:
            TofPoints = 5000
        tmp2 = int(TofPoints/tmp)

        Data1Reshaped = np.reshape(data1,(64,250,tmp,tmp2))
        Data1Reshaped2 = np.sum(Data1Reshaped,axis = 3)#/self.CosAlpha[:,None]
        Data2Reshaped = np.reshape(data2,(64,250,tmp,tmp2))
        Data2Reshaped2 = np.sum(Data2Reshaped,axis = 3)#/self.CosAlpha[:,None]
        DataStacked = np.vstack((Data2Reshaped2,Data1Reshaped2))

        FallingPixels = self.FallingDistance/self.TubeHeight
        DataStacked = falling_correction(DataStacked,FallingPixels)

        try:
            effic1 = np.load(r'./npyfiles/D3_effic1_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',allow_pickle=True)
        except:
            effic1 = self.efficiency_matrix()

        #print(np.sum(DataStacked))
        #effic1 = self.efficiency_matrix()
        DataStacked = DataStacked/effic1
        #print(np.average(effic1))

        if int(RunNum[-5:]) >= 20810 and int(RunNum[-5:]) <= 20813:
             DataTmp1= DataStacked[:,:,180:250].copy()
             DataTmp2 = DataStacked[:,:,0:180].copy()
             DataStacked[:,:,0:70] = DataTmp1
             DataStacked[:,:,70:250] = DataTmp2

        if self.MaskSwitch == True:
            mask = np.load(r'masks/D3Mask.npy',allow_pickle=True)
            DataStacked = DataStacked*mask[:,:,None]   
        path1 = r'./npyfiles/D3GroupX' + str(round(self.XCenter,2)) + '-' + str(round(self.YCenter,2)) + '_'+ str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.pkl'
        path2 = r'./npyfiles/D3GroupY' + str(round(self.XCenter,2)) + '-' + str(round(self.YCenter,2)) + '_'+ str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.pkl'
        load_successful = False
        if os.path.exists(path1) and os.path.exists(path2):
            try:
                with open(path1, 'rb') as f:
                    GroupX = pickle.load(f)
                with open(path2, 'rb') as f:
                    GroupY = pickle.load(f)
                load_successful = True
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                print(f"Error loading D3 pickle files: {e}. Regenerating GroupX and GroupY...")
                # 删除损坏的文件
                try:
                    os.remove(path1)
                    os.remove(path2)
                except:
                    pass

        if not load_successful:
            GroupX,GroupY = self.detector_group(Height,Width,R,XBins,YBins,RBins,XCenter,YCenter)
        #D3_effeciency = np.load(r'./npyfiles/normMatrix.npy',allow_pickle=True)
        #DataStacked = DataStacked/D3_effeciency[:,None]
        return DataStacked,ProtonCharge,GroupX,GroupY
        

    def direct_beam_integrate_to_lambda(self,data):
        #data,_ = self.load_data(RunNum)
        xx = self.WavelengthArray2
        xp = self.WavelengthArray
        yp = np.zeros_like(data)
        for i in range(len(xx)):        
            f = interpolate.interp1d(xx[i], data[i],fill_value="extrapolate",bounds_error = False)
#            f = interpolate.interp1d(np.hstack((xx[i]-self.TOF,xx[i],xx[i]+self.TOF)), np.hstack((np.zeros_like(data[i]),data[i],np.zeros_like(data[i]))),fill_value="extrapolate",bounds_error = False) 
            yp[i] = f(xp)
        return xp,np.sum(yp,axis = 0)

    def direct_beam_integrate_to_lambda2(xp, self,RunNum):
        data,_ = self.load_data(RunNum)
        xx = self.WavelengthArray2
        #xp = self.WavelengthArray
        yp = np.zeros_like(data)

        for i in range(len(xx)):
            f = interpolate.interp1d(xx[i], data[i],fill_value="extrapolate",bounds_error = False)
#            f = interpolate.interp1d(np.hstack((xx[i]-self.TOF,xx[i],xx[i]+self.TOF)), np.hstack((np.zeros_like(data[i]),data[i],np.zeros_like(data[i]))),fill_value="extrapolate",bounds_error = False) 
            yp[i] = f(xp)
        return xp,np.sum(yp,axis = 0)


    def integrate_x(self,data):
        
        output = np.sum(data,axis = 0)
        return output  

#    def integrate_lambda(self,data,LambdaMin = 0,LambdaMax = 5000):
#        output = np.sum(data[LambdaMin:LambdaMax,:],axis = 1)
#        return output  

    def delta_q_calc(self,wavelength,r):
        '''Calculate the resolution function array, 
        input: wavelength array and radius array, 
        output: a two dimentional array with the corresponding sigma Q of the gaussian resolution in the cells'''

        LP = 1/(1/self.L1 +1/self.L2)
        delta_q = 1/12*(2*np.pi/wavelength)*(3*(self.A1/2)**2/self.L1**2 + 3*(self.A2/2)**2/LP**2 +\
                        (self.TubeWidth**2+self.TubeHeight**2)/self.L2**2 + r[:,None]**2/self.L2**2*(self.DeltaLambdaRatio)**2)
        return delta_q

    def q_calc(self,wavelength,r):
        '''Calculate the Q array, 
        input: wavelength array and radius array, 
        output: a two dimentional array with the corresponding Q in the cells'''
        theta = np.arctan(r[:,None]/self.L2)
        q = 4*np.pi*np.sin(theta/2)/wavelength        
        return q
    
    def find_x_center(self,data):
        DataTimeIntegrated = np.sum(data,axis = 1)
        center = list(DataTimeIntegrated).index(np.max(DataTimeIntegrated))
        return center
    
    def zero_divide(self,a,b):
        c = np.divide(a,b,out = np.zeros_like(a), where = b!=0)
        return c

    def moving_average(self,interval, WindowSize = 10):
        window = np.ones(int(WindowSize)) / float(WindowSize)
        output = np.convolve(interval, window, 'same')
        return output
        
    def linear_func(self,k,b,x):
        return k*x+b
    
    def trans_calc(self,data1,data2):
        StartFit = 20
        StopFit = int(len(data1) - 20)
        Data1Smoothed = self.moving_average(data1,WindowSize = 3)
        Data2Smoothed = self.moving_average(data2,WindowSize = 3)
        Div = self.zero_divide(Data1Smoothed,Data2Smoothed)
        X = np.linspace(StartFit,StopFit,StopFit-StartFit)
        Y = Div[StartFit:StopFit]
        c, stats = P.polyfit(X,Y,3,full=True)
        #paras,err=scipy.optimize.curve_fit(lambda k,b,X: self.linear_func(k,b,X),X,Y,method = 'dogbox')
        DataX = np.arange(len(data1))
        DivFitedY = c[0] + c[1]*DataX + c[2]*DataX**2 + c[3]*DataX**3
        return DivFitedY      

    def q_dict_generate(self,D3QX):
        QDict = {}
        QX = self.QX2D
        QY = self.QY2D
        QArray = self.QArray2d
        for i in range(self.XBins):
            for j in range(self.YBins):
                for k in self.WaveBinsSelected:
                #for k in range(self.WaveBins):
                    Tmp1 = self.QBins2D - len(QX[QX > D3QX[i,k]])-1
                    Tmp2 = self.QBins2D - len(QY[QY > self.D3QY[j,k]])-1
                    if (Tmp2,Tmp1) in QDict.keys():
                        QDict[(Tmp2,Tmp1)].append([i,j,k])
                    else:
                        QDict[(Tmp2,Tmp1)] = []
                        QDict[(Tmp2,Tmp1)].append([i,j,k])
        np.save('./npyfiles/QDictFile' + str(self.WaveMin) + '-' + str(self.WaveMax)  + '_' + str (QX[-1]) + str(self.QBins2D) + '.npy',QDict)
        return QDict


    def translate_to_q_2d(self,I0Smoothed,Sample2DTransNormed,Cell2DTransNormed,AirDirectPC,SampleScatteringPC,CellScatteringPC):
        global lower
        global upper
        lower = self.StartWave
        bins = 1.0
        upper = self.StopWave

        Height = self.BankHeight
        Width = self.BankWidth
        R = self.R
        RBins = self.RBins
        XBins = self.XBins
        YBins = self.YBins
        XCenter = self.XCenter
        YCenter = self.YCenter
        QX = self.QX
        QY = self.QY
        QYNorm = np.zeros(len(QY))
        QYCell = np.zeros(len(QY))
        DeltaQ = np.zeros(len(QY))
        QSquare = np.zeros(len(QY))
        QAve = np.zeros(len(QY))
        CountSum = np.zeros(len(QY))
        WavelengthArray = self.WavelengthArray
        #WaveBins = len(SampleTransNormed[0])
        #RBins = len(SampleTransNormed)

        #SampleTransNormed2 = SampleTransNormed#/DLambda
        #RArray = self.RArray #np.arange(0,self.R,self.R/self.RBins)
        I0Array = np.average(I0Smoothed,axis = 0) #np.sum(I0Smoothed,axis = 0) #*np.ones(len(SampleTransNormed))[:,None]*np.ones(len(WavelengthArray))
        #solid_angle_2d(self,Height,Width,R,XBins,YBins,XCenter,YCenter)
        SolidAngle = self.solid_angle_2d(Height,Width,R,XBins,YBins,XCenter,YCenter)
        #SolidAngle = self.R/self.RBins*2*np.pi*RArray[:,None]/self.L2**2*WavelengthArray/WaveBins
        #NormData2 = self.zero_divide(SampleTransNormed,I0Array)
        Normalization = SolidAngle[:,:,None]*I0Array*self.SampleThickness/10*self.A2**2/self.A2Small**2
        #plt.plot(np.sum(NormData2,axis = 0)); plt.yscale('log')
        #plt.plot(np.sum(I0Array,axis = 0))
        Normalization = Normalization/bins
        #SampleTransNormed = self.zero_divide(SampleTransNormed,Normalization)
        #ThetaArray = np.arctan(RArray/self.L2)
        #QArray = 4*np.pi*np.sin(ThetaArray[:,None]/2)/WavelengthArray
        #DeltaQArray = self.delta_q_calc(WavelengthArray[50],RArray)
        Sample2DTransNormed = self.denan_2d(Sample2DTransNormed)
        Cell2DTransNormed = self.denan_2d(Cell2DTransNormed)
        #print(Sample2DTransNormed[Sample2DTransNormed == np.nan])
        #print(Cell2DTransNormed[Cell2DTransNormed==np.nan]) 
        #Sample2DTransNormedLeft = np.zeros_like(Sample2DTransNormed)
        #Sample2DTransNormedRight = np.zeros_like(Sample2DTransNormed)
        #Sample2DTransNormedLeft[1:] = Sample2DTransNormed[1:]*2/3 + Sample2DTransNormed[:-1]*1/3
        #Sample2DTransNormedLeft[0] = Sample2DTransNormed[0]
        #Sample2DTransNormedRight[:-1] = Sample2DTransNormed[:-1]*2/3 + Sample2DTransNormed[1:]*1/3
        #Sample2DTransNormedRight[-1] = Sample2DTransNormed[-1]
 
        QX2D,QY2D,QZ2DCenter,QZ2DError = self.bining_2d(self.D3QX,Sample2DTransNormed,Cell2DTransNormed,Normalization,AirDirectPC,SampleScatteringPC,CellScatteringPC)
        #_,_,QZ2DLeft,_ = self.bining_2d(self.D3QXLeft,Sample2DTransNormedLeft,Cell2DTransNormed,Normalization)
        #_,_,QZ2DRight,_ = self.bining_2d(self.D3QXRight,Sample2DTransNormedRight,Cell2DTransNormed,Normalization)
        #print(QZ2DLeft[QZ2DLeft>0],'Left')
        
        QZ2D = QZ2DCenter #(QZ2DCenter + QZ2DLeft + QZ2DRight)/3
        #print(QZ2DCenter[QZ2DCenter>0],'QZ')
        return QX2D,QY2D,QZ2DCenter,QZ2DError

    def bining_2d(self,D3QX,Sample2DTransNormed,Cell2DTransNormed,Normalization,AirDirectPC,SampleScatteringPC,CellScatteringPC):
        QArray2dSample = self.QArray2d.copy()
        QArray2dCell = self.QArray2d.copy()
        QArrayNorm0 = self.QArray2d.copy()
        try:
            QDictLoaded = np.load(r'./npyfiles/QDictFile' + str(self.WaveMin) + '-' + str(self.WaveMax) + '_' + str (round(D3QX[-1][-1],3)) + str(self.QBins2D) + '.npy',allow_pickle=True).item()
            
        except: 
            QDictLoaded = self.q_dict_generate(D3QX)

        #print('Length of the Dict is ' + str(len(QDictLoaded)))
        for key, items in QDictLoaded.items():
            items = np.array(items)
            QArray2dSample[key[0],key[1]] = np.sum(Sample2DTransNormed[items[:,0],items[:,1],items[:,2]])
            QArray2dCell[key[0],key[1]] = np.sum(Cell2DTransNormed[items[:,0],items[:,1],items[:,2]])
            QArrayNorm0[key[0],key[1]] = np.sum(Normalization[items[:,0],items[:,1],items[:,2]])
        QX2D = self.QX2D
        QY2D = self.QY2D
        QArrayNorm = QArrayNorm0
        #valv = int(self.QBins2D/4)
        #QArrayNorm = (QArrayNorm0 + np.hstack((QArrayNorm0[:,1:],QArrayNorm0[:,0][:,None])) + np.hstack((QArrayNorm0[:,-1][:,None],QArrayNorm0[:,:-1])))/3  
        #QArrayNorm[valv:valv*(-1)] = QArrayNorm0[valv:valv*(-1)]
        QZ2D = (QArray2dSample/SampleScatteringPC - QArray2dCell/CellScatteringPC)/(QArrayNorm/AirDirectPC)
        QZ2D = self.denan(QZ2D)
        QZ2DError = np.sqrt((np.sqrt(QArray2dSample)/SampleScatteringPC)**2 + (np.sqrt(QArray2dCell)/CellScatteringPC)**2)/(QArrayNorm/AirDirectPC)
        QZ2DError = self.denan(QZ2DError)
        return QX2D,QY2D,QZ2D,QZ2DError

    def add_error(self,deltax,deltay):
        return np.sqrt((deltax)**2 + (deltay)**2)
 
    def divide_error(self,deltax,x,deltay,y):
        return x/y*np.sqrt((deltax/x)**2 + (deltay/y)**2) 

    def inelastic_correction(self,data,ThetaArray,WavelengthArray,I0Norm):
        corrector = claude_correction.InelasticCorrector(
            data=data,
            theta_array=ThetaArray,
            lambda_array=WavelengthArray,
            i0_lambda=I0Norm,
            n_q_bins=120,
            q_min=None,
            q_max=None
        )
        result = corrector.run(
        n_iterations=15,
        damping=0.5,
        weight_exponent=0.5,
        convergence_threshold=1e-4,
        min_counts=1.0,
        verbose=False
        )
        return result['D']
        

    def translate_to_q(self,I0Smoothed,AirDirectPC,I0Scale,SampleTransNormed,SamplePC,SampleTrans,CellTransNormed,CellPC,CellTrans):
        global lower
        global upper
        lower = self.StartWave
        bins = 1.0
        upper = self.StopWave
      
        Height = self.BankHeight
        Width = self.BankWidth
        R = self.R
        RBins = self.RBins
        XBins = self.XBins
        YBins = self.YBins
        XCenter = self.XCenter
        YCenter = self.YCenter
        QX = self.QX
        if QX[4]-QX[3] == QX[3]-QX[2]:
            QXHalfBin = (QX[1:] + QX[:-1])/2
        else:
            QXHalfBin = np.sqrt(QX[1:]*QX[:-1])
        QY = self.QY
        QYNorm = np.zeros(len(QY))
        QYCell = np.zeros(len(QY)) 
        DeltaQ = np.zeros(len(QY))
        QSquare = np.zeros(len(QY))
        QAve = np.zeros(len(QY))
        CountSum = np.zeros(len(QY))
        WavelengthArray = self.WavelengthArray2
        WaveBins = len(SampleTransNormed[0])
        RBins = len(SampleTransNormed)

        #SampleTransNormed2 = SampleTransNormed #/DLambda
        RArray = self.RArray #np.arange(0,self.R,self.R/self.RBins)
        I0Array = I0Smoothed#*np.ones(len(SampleTransNormed))[:,None]*np.ones(len(WavelengthArray))
        SolidAngle = self.solid_angle(Height,Width,R,XBins,YBins,RBins,XCenter,YCenter)
        #SolidAngle = self.R/self.RBins*2*np.pi*RArray[:,None]/self.L2**2*WavelengthArray/WaveBins
        #NormData2 = self.zero_divide(SampleTransNormed,I0Array)
        Normalization = SolidAngle[:,None]*I0Array*self.SampleThickness/10*self.A2**2/self.A2Small**2
        #plt.plot(np.sum(NormData2,axis = 0)); plt.yscale('log')
        #plt.plot(np.sum(I0Array,axis = 0))
        Normalization = Normalization/bins #*I0Scale
        #SampleTransNormed = self.zero_divide(SampleTransNormed,Normalization)
        ThetaArray = np.arctan(RArray/self.L2)
        QArray = 4*np.pi*np.sin(ThetaArray[:,None]/2)/WavelengthArray
        DeltaQArray = self.delta_q_calc(WavelengthArray[int(self.RBins/2)],RArray)
       
        #bin_edges, bin_indices = yuanbao.create_q_bins(QArray, num_bins=100)
        #DLambda, SampleTransNormed = yuanbao.optimize_d_with_weighting(SampleTransNormed, QArray, bin_indices,Normalization,len(bin_edges)-1) 
        #self.save_matrix(DLambda,'DLambda.dat')
        #np.save('SampleTransNormed.npy',SampleTransNormed)
        #np.save('ThetaArray.npy',ThetaArray)
        #np.save('WavelengthArray.npy',self.WavelengthArray)
        #np.save('Normalization.npy',Normalization)
        #DLambda = self.inelastic_correction(SampleTransNormed,ThetaArray,self.WavelengthArray,Normalization)
        #DLambda = np.load('DLambda.npy',allow_pickle=True)
        #SampleTransNormed = SampleTransNormed*DLambda
        #np.save('DLambda.npy',DLambda) 
        for i in np.arange(self.BeamStopStart,RBins):  #(5,RBins):
            for j in self.WaveBinsSelected: #np.arange(lower,upper): #(0,WaveBins)
                tmp = self.QBins - len(QXHalfBin[QXHalfBin > QArray[i,j]]) - 1 #len(QX) - len(QX[QX > QArray[i,j]])#
                CountIJ = SampleTransNormed[i,j]-CellTransNormed[i,j]
                QY[tmp] += SampleTransNormed[i,j]#/RArray[i]*5*WavelengthArray[j]
                QYCell[tmp] += CellTransNormed[i,j]#/RArray[i]*5*WavelengthArray[j]
                QYNorm[tmp] += Normalization[i,j]
                DeltaQ[tmp] += DeltaQArray[i,j]*CountIJ
                QSquare[tmp] += QArray[i,j]**2*CountIJ
                QAve[tmp] += QArray[i,j]*CountIJ
                CountSum[tmp] += CountIJ

        GroupingScale = 1/2
        QYError = self.zero_divide(np.sqrt((np.sqrt(QY*np.average(SampleTrans)/GroupingScale)/SamplePC*self.SampleScatteringFactor)**2+(np.sqrt(QYCell*np.average(CellTrans)/GroupingScale)/CellPC*self.CellScatteringFactor)**2),QYNorm/AirDirectPC)
        QYSampleCellError = self.zero_divide((np.sqrt(QY*np.average(SampleTrans)/GroupingScale)/SamplePC*self.SampleScatteringFactor),QYNorm/AirDirectPC)
        QYCellError = self.zero_divide((np.sqrt(QYCell*np.average(CellTrans)/GroupingScale)/CellPC*self.CellScatteringFactor),QYNorm/AirDirectPC)

        QYSample = self.zero_divide((QY/SamplePC*self.SampleScatteringFactor-QYCell/CellPC*self.CellScatteringFactor),QYNorm/AirDirectPC)#self.zero_divide((QY-QYCell),QYNorm)
        QYSampleCell = self.zero_divide((QY/SamplePC*self.SampleScatteringFactor),QYNorm/AirDirectPC)
        QYCell = self.zero_divide((QYCell/CellPC*self.CellScatteringFactor),QYNorm/AirDirectPC)

        QXError = np.sqrt(np.abs(self.zero_divide(DeltaQ,CountSum) + self.zero_divide(QSquare,CountSum) - (self.zero_divide(QAve,CountSum))**2))
        QXError = self.denan(QXError)
        #QXError = (DeltaQ/CountSum + QSquare/CountSum - (QAve/CountSum)**2)
        return QX,QYSample,QYError,QXError,QYSampleCell,QYSampleCellError,QYCell,QYCellError



    def data_plot_xy(self,xx,yy,label='Lambda plot', show = True,FileName='LambdaPlot'):
        plt.plot(xx,yy,label = label)
        #plt.title(SampleName)
        #plt.xlim(1,15)
        plt.legend(fancybox=True, framealpha=0.01,frameon = False)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Counts (n/s/Å)')
        #plt.yscale('log')
        plt.savefig(FileName + '.svg',dpi = 600, format = 'svgz',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
        plt.close()

    def data_plot_xyz(self,xx,yy,zz,label='Lambda plot',label2 = 'LambdaPlot', show = True,FileName='LambdaPlot'):
        plt.plot(xx,yy,label = label)
        plt.plot(xx,zz,label = label2)
        #plt.title(SampleName)
        plt.ylim(0,np.max(yy)*1.2)
        plt.legend(fancybox=True, framealpha=0.01,frameon = False)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Transmission (n/s/Å)')
        #plt.yscale('log')
        plt.savefig(FileName + '.svg',dpi = 600, format = 'svgz',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
        plt.close()

    def x_plot(self,data):
        XBins = len(data)
        XBin = self.DetectorWidth/XBins
        XArray = np.arange(-1*self.DetectorWidth/2+XBin/2,self.DetectorWidth/2,XBin)
        XIntensity = np.sum(data,axis = 1)
        plt.plot(XArray,XIntensity)
        plt.yscale('log')
        #myplot_linear.plot(XArray,XIntensity,xlabel = 'X (mm)', ylabel = 'Intensity', save = True, save_name = 'XPlot.svg')

    def save_file(self,xx,yy,zz,file_name):
        with open(file_name,'w') as f:
            print('{:<20.12s}{:<20.12s}{:<20.12s}'.format('Q','I(Q)','Sigma I(Q)'),file = f)
            for x,y,z in zip(xx,yy,zz):
                print('{:<20.8f}{:>20.8f}{:>20.8f}'.format(x,y,z),file = f)
        f.close()

    def save_file4(self,xx,yy,zz,tt,file_name,ExperimentTime):
        tt = [x if isinstance(x, (int, float)) else 0 for x in tt]
        with open(file_name,'w') as f:
            print('Problems or bugs please contact Taisen Zuo (zuots@ihep.ac.cn) ' + 'Experiment started at ' + str(ExperimentTime[0]) + ' and run ' + str(round(ExperimentTime[3],2)) + ' minutes. ' + 'File created:' + str(self.formated_time),file = f)
            print('{:<20.12s}{:<20.12s}{:<20.12s}{:<20.12s}'.format('Q','I(Q)','Sigma I(Q)','Sigma Q'),file = f)
            for x,y,z,t in zip(xx,yy,zz,tt):
                print('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(x,y,z,t),file = f)
        f.close()


    def save_file40(self,xx,yy,zz,tt,file_name):
        with open(file_name,'w') as f:
            print('{:<20.12s}{:<20.12s}{:<20.12s}{:<20.12s}'.format('Q','I(Q)','Sigma I(Q)','Sigma Q'),file = f)
            for x,y,z,t in zip(xx,yy,zz,tt):
                print('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(x,y,z,t),file = f) 
        f.close()

    def save_file4_QXY(self,xx,yy,zz,tt,file_name):
        with open(file_name,'w') as f:
            print('Data columns Qx - Qy - I(Qx,Qy) - err(I)')
            print('ASCII data')      
            #print('{:<20.12s}{:<20.12s}{:<20.12s}{:<20.12s}'.format('QX','QY','I(Q)','Sigma I(Q)'),file = f)
            for x,y,z,t in zip(xx,yy,zz,tt):
                print('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(x,y,z,t),file = f)
        f.close()

    def save_file1(self,xx,file_name):
        with open(file_name,'w') as f:
            for x in xx:
                print('{:<20.8f}'.format(x),file = f)
        f.close()

    def save_file2(self,xx,yy,file_name):
        with open(file_name,'w') as f:
            for x,y in zip(xx,yy):
                print('{:<20.8f}{:>20.8f}'.format(x,y),file = f)
        f.close()

    def save_file3(self,xx,yy,zz,file_name):
        with open(file_name,'w') as f:
            for x,y,z in zip(xx,yy,zz):
                print('{:<20.8f}{:>20.8f}{:>20.8}'.format(x,y,z),file = f)
        f.close()

    def save_matrix(self,matrix,FileName):
        with open(FileName,'w') as f:
            for i in matrix:
                for j in i:
                    print('{:<20.8f}'.format(j),file = f, end = '')
                print('\n',file = f, end = '')
        f.close()


    def get_variable_name(self,p):
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',line)
            if m:
                return m.group(1)

    def save_image(self,xx,yy,ImageName):
        myplot_log_log.plot(xx,yy,xlabel = 'Q (Å$^{-1}$)', ylabel = 'I (cm$^{-1}$)', save = True, save_name = ImageName)

    def save_image_with_error_bar(self,xx,yy,ErrorBar,ImageName,label):
        myplot222.plot(xx,yy,ErrorBar,xlabel = 'Q (Å$^{-1}$)', ylabel = 'I (cm$^{-1}$)', label = label,save = True, save_name = ImageName)

    def I0_interp(self, data,func):
        out = np.zeros_like(data)
        for i in range(len(data)):
            out[i] = func(data[i])
        return out

    def denan(self,arr):
        for i in range(len(arr)):
            if arr[i] is None or (isinstance(arr[i], float) and np.isnan(arr[i])):
                arr[i] = 0
        return arr


    def denan_2d(self,data):
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] is None or (isinstance(data[i][j], float) and np.isnan(data[i][j])):
                    data[i][j] = 0
        return data

    def plot_data_2d(self,QX,QY,QArray,show = True,FileName = '2D_plot', logscale = False):
        if logscale == True:
            #QArray[QArray == np.nan] = 1E-10
            #QArray[QArray <= 0] = 1E-10
            QArray = np.log(np.abs(QArray.copy()))
        #plt.contour(QX,QY,QArray,600,cmap = 'hot')
        plt.figure(figsize=(5, 5))
        plt.pcolormesh(QX,QY,QArray,cmap='jet',shading='auto')
        #cbar = plt.colorbar(im, label='Intensity')
        #plt.figure(figsize=(5, 5)) 
        plt.xlabel(r'QX ($\AA^{-1}$)',fontsize=12)
        plt.ylabel(r'QY ($\AA^{-1}$)',fontsize=12)
        plt.tight_layout()  # 自动调整布局，避免元素重叠
        #plt.colorbar(label='Values')
        plt.savefig(FileName + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
        plt.close()
    
    def mat_plot_2d(self,QArray,show = True,FileName = 'MatPlot2D', logscale = False):
        if logscale == True:
            #QArray[QArray == np.nan] = 1E-10
            #QArray[QArray <= 0] = 1E-10
            #QArray = QArray + 1E-8
            QArray = np.log(QArray.copy())
        plt.figure(figsize=(5, 5))
        ax = plt.matshow(QArray,cmap = plt.cm.jet)
        plt.xlabel(r'QX ($\AA^{-1}$)')
        plt.ylabel(r'QY ($\AA^{-1}$)')
        plt.colorbar(label='Values')
        plt.savefig(FileName + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
        plt.close()

    def read_exp_data(self,FileName):
        with open(FileName,"r") as f:
            qqx = []
            sio2iq = []
            QyError = []
            for line in f.readlines():
                sline = line.split()
                qqx.append(float(sline[0]))
                sio2iq.append(float(sline[1]))
                QyError.append(float(sline[2]))
        return qqx,sio2iq,QyError

def zero_divide(a,b):
    c = np.divide(a,b,out = np.zeros_like(a), where = b!=0)
    return c


def reduce(samples,info):
    global data_reduce
    global data_reduce0
    DataFold = info.DataFold
    #RunInfoFold = info.RunInfoFold
    data_reduce = data_reduce(DataFold,  info)
    data_reduce0 = data_reduce_D3.data_reduce(DataFold,info)
    SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness = samples
    reduce_one_data(data_reduce, SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness,info)
#    return output

def reduce_2d(samples,info):
    global data_reduce
    global data_reduce0
    DataFold = info.DataFold
    #RunInfoFold = info.RunInfoFold
    data_reduce = data_reduce(DataFold, info)
    data_reduce0 = data_reduce_D3.data_reduce(DataFold,info)
    SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness = samples
    #reduce_one_data(data_reduce, SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness,info)
    reduce_one_data_2d(data_reduce, SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness,info)

def get_run_fold(run_num):
    run_fold = r"RUN" + str('0'*(7-len(run_num.split('_')[0]))) + run_num
    return run_fold

def reduce_one_data_2d(data_reduce, SampleName,SampleScattering,CellScattering,SampleDirect,CellDirect,AirDirect,SampleThickness,info):
    global data_reduce0
    SampleScatteringRun = get_run_fold(SampleScattering) #r"RUN" + str('0'*(7-len(SampleScattering))) + SampleScattering
    SampleDirectRun =  get_run_fold(SampleDirect) #r"RUN"  + str('0'*(7-len(SampleDirect))) + SampleDirect
    CellScatteringRun =  get_run_fold(CellScattering) #r"RUN"  + str('0'*(7-len(CellScattering))) + CellScattering
    CellDirectRun =  get_run_fold(CellDirect) #r"RUN"  + str('0'*(7-len(CellDirect))) + CellDirect
    AirDirectRun =  get_run_fold(AirDirect) #r"RUN"  + str('0'*(7-len(AirDirect))) + AirDirect
    data_reduce.SampleThickness = float(SampleThickness)

    AirDirect,AirDirectPC = data_reduce0.load_data(AirDirectRun)
    CellDirect,CellDirectPC = data_reduce0.load_data(CellDirectRun)
    SampleDirect,SampleDirectPC = data_reduce0.load_data(SampleDirectRun)
    #AirDirect = AirDirect/AirDirectPC
    #CellDirect = CellDirect*info.DirectFactorCell/CellDirectPC
    #SampleDirect = SampleDirect*info.DirectFactorSample/SampleDirectPC

    CellScattering,CellScatteringPC = data_reduce.load_data(CellScatteringRun)#*0
    SampleScattering,SampleScatteringPC = data_reduce.load_data(SampleScatteringRun)

    SampleScattering2D,ProtonCharge,_,_ = data_reduce.load_data_origin(SampleScatteringRun)
    #SampleScattering2D = SampleScattering2D/ProtonCharge*9000*390701
    CellScattering2D,ProtonCharge,_,_ = data_reduce.load_data_origin(CellScatteringRun)
    #CellScattering2D = CellScattering2D/ProtonCharge*9000*390701

    I0X,I0Y = data_reduce.direct_beam_integrate_to_lambda(AirDirect)
    ISX,ISY = data_reduce.direct_beam_integrate_to_lambda(SampleDirect)
    ICX,ICY = data_reduce.direct_beam_integrate_to_lambda(CellDirect)
    SampleTrans = data_reduce.trans_calc(ISY/SampleDirectPC,I0Y/AirDirectPC)
    CellTrans = data_reduce.trans_calc(ICY/CellDirectPC,I0Y/AirDirectPC)
    Sample2DTransNormed = SampleScattering2D/SampleTrans
    Cell2DTransNormed = CellScattering2D/CellTrans
    #SampleTrans = data_reduce.trans_calc(data_reduce.integrate_x(SampleDirect), data_reduce.integrate_x(AirDirect))
#    data_reduce.save_file2(I0X,I0Y,info.OutPath + '/' + "I_Air_Lambda" + SampleName + ".dat")
#    data_reduce.save_file2(ISX,ISY,info.OutPath + '/' + "I_Sample_Lambda" + SampleName + ".dat")
#    data_reduce.save_file2(ICX,ICY,info.OutPath + '/' + "I_Cell_Lambda" + SampleName + ".dat")
    #CellTrans = data_reduce.trans_calc(data_reduce.integrate_x(CellDirect), data_reduce.integrate_x(AirDirect))

    SampleTransNormed = zero_divide(SampleScattering, SampleTrans)
    #CellTransNormed =  zero_divide(CellScattering,CellTrans) # 
    #I0X,I0Y = data_reduce.direct_beam_integrate_to_lambda(AirDirect)
   # plt.plot(I0X,I0Y)#plot the I0
    I0 = I0Y*data_reduce.I0Scale
    I0Smoothed = data_reduce.moving_average(I0,5)
    I0Y = I0Smoothed
#    I0TofArray = np.linspace(data_reduce.TimeDelay,data_reduce.TimeDelay + data_reduce.TOF,data_reduce.WaveBins)
#    I0WavelengthArray = data_reduce.const*I0TofArray/(data_reduce.SamplePos + data_reduce.L2)
    I0X = np.hstack((I0X - data_reduce.TOF, I0X, I0X + data_reduce.TOF))
    I0Y = np.hstack((np.zeros_like(I0Y),I0Y,np.zeros_like(I0Y)))
    #f = interpolate.interp1d(I0X, I0Y,fill_value="extrapolate",bounds_error = False)
    #I0Interpolated = data_reduce.I0_interp(data_reduce.WavelengthArray2,f)
    I0Interpolated = np.interp(data_reduce.WavelengthArray2,I0X,I0Y,left = 0, right = 0)

    if info.debug is True and info.debug2 is True:
        SampleSumed = np.sum(np.sum(SampleTransNormed[45:55],axis = 0))
        Sample50Sumed = np.sum(np.sum(SampleTransNormed[0:30],axis = 0))
        Sample99Sumed = np.sum(np.sum(SampleTransNormed[89:99],axis = 0))
        I0Sumed = np.sum(I0Interpolated[50]/10)


        plt.plot(data_reduce.WavelengthArray2[50],I0Interpolated[50]/10,label = 'I(Lambda)AirDirect')
        plt.plot(data_reduce.WavelengthArray2[50],I0Sumed/SampleSumed*np.sum(SampleTransNormed[45:55],axis = 0),label = 'I(Lambda)SampleScatteringDmiddle')
        plt.plot(data_reduce.WavelengthArray2[0],I0Sumed/Sample50Sumed*np.sum(SampleTransNormed[0:30],axis = 0),label = 'I(Lambda)SampleScatteringD3_Rmin')
        plt.plot(data_reduce.WavelengthArray2[99],I0Sumed/Sample99Sumed*np.sum(SampleTransNormed[89:99],axis = 0),label = 'I(Lambda)SampleScatteringD3_Rmax')

        plt.title(SampleName)
        #plt.xlim(1,15)
        plt.legend(fancybox=True, framealpha=0.01,frameon = False)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Counts (n/s/Å)')
        plt.yscale('log')
        plt.show()
        plt.close()
     #
        #print('I_SampleDirect_D3:',np.sum(data_reduce.load_data(SampleDirectRun)))
        #print('I_SampleScattering_D3:',np.sum(data_reduce.load_data(SampleScatteringRun)))


    QX2D,QY2D,QZ2D,QZError2D = data_reduce.translate_to_q_2d(I0Interpolated,Sample2DTransNormed,Cell2DTransNormed,AirDirectPC,SampleScatteringPC,CellScatteringPC)
    #QZ2D = np.zeros_like(QZ2D0)
    #print(QZ2D0.shape)
    #QZ2D = (QZ2D0 + np.hstack((QZ2D0[:,1:],QZ2D0[:,0][:,None])) + np.hstack((QZ2D0[:,-1][:,None],QZ2D0[:,:-1])))/3
    #QZ2D = QZ2D + 1E-15
    #data_reduce.mat_plot_2d(np.sum(Sample2DTransNormed,axis = 2),show = True)
#    data_reduce.mat_plot_2d(QZ2D,show = info.debug, FileName = info.OutPath + '/' + "Mat2DPlot" + SampleName)
    path = info.OutPath + '/' 
    if not os.path.exists(path + 'LogPlots'):
        os.system("mkdir " + path + 'LogPlots')
    if not os.path.exists(path + 'MatPlots'):
        os.system("mkdir " + path + 'MatPlots')
    if not os.path.exists(path + 'LinearPlots'):
        os.system("mkdir " + path + 'LinearPlots')
    #data_reduce.mat_plot_2d(QZ2D.copy(),show = info.debug, FileName = info.OutPath + '/'+ 'MatPlots' +'/' + "Mat2DPlot_logscale_" + SampleName,logscale = info.DataReduce2DScaleLog)

    #print(QZ2D[QZ2D>0])
#    data_reduce.plot_data_2d(QX2D,QY2D,QZ2D,show= info.debug, FileName = info.OutPath + '/' + "QXQY2DPlot" + SampleName)
    #data_reduce.plot_data_2d(QX2D,QY2D,QZ2D.copy(),show= info.debug, FileName = info.OutPath + '/'+ 'LogPlots' +'/' + "QXQY2DPlot_logscale_" + SampleName, logscale = info.DataReduce2DScaleLog)
    #data_reduce.plot_data_2d(QX2D,QY2D,QZ2D.copy(),show= info.debug, FileName = info.OutPath + '/' + 'LinearPlots' + '/'+ "QXQY2DPlot_linearscale_" + SampleName, logscale = False)   
    #QYBool = (QY2D > 0.002)*(QY2D < 0.01)
    #QZselect = QZ2D*QYBool #[np.ones_like(QX2D)*QYBool[:,None]]
    #print(QYBool)
    #for i in (QZselect):
    #    print(i)
    if info.GISANS_mode == True:
        outfile2 = info.OutPath + '/' + "QY" + str(info.GIQYMin) + '-' + str(info.GIQYMax) + "Selected" + SampleName + ".dat" 
        f = open(outfile2,'w')
        k = 0
        #print(QZ2D.shape)
        #print(QX2D.shape)
        #print(QY2D.shape)
        tt = []
        for i in range(len(QZ2D)):
            if QY2D[i] > info.GIQYMin and QY2D[i] < info.GIQYMax:
                tt.append([])
                for j in range(len(QZ2D[0])):
                    tt[k].append(QZ2D[i,j])
                k += 1
        tt = np.sum(np.array(tt),axis = 0)
        for i in range(len(tt)):
            print(QX2D[i],tt[i],file = f)
        f.close()

    if info.GISANS_mode == True:
        outfile3 = info.OutPath + '/' + "QX" +str(info.GIQXMin) + '-' + str(info.GIQXMax) + "Selected" + SampleName + ".dat"
        f = open(outfile3,'w')
        k = 0

        tt = []
        for i in range(len(QZ2D[0])):
            if QX2D[i] > info.GIQXMin and QX2D[i] < info.GIQXMax:
                tt.append([])
                for j in range(len(QZ2D)):
                    tt[k].append(QZ2D[j,i])
                k += 1
        tt = np.sum(np.array(tt),axis = 0)
        for i in range(len(tt)):
            print(QY2D[i],tt[i],file = f)
        f.close()
 
    #data_reduce.save_file2(QX2D,np.sum(QZselect,axis = 0),info.OutPath + '/' + "QY0.002-0.01Selected" + SampleName + ".dat")
    #QXP,QYP,QYError,QXError = data_reduce.translate_to_q(I0Interpolated,SampleTransNormed,CellTransNormed)
    ###data_reduce.save_image(QXP,QYP,"IQ_Normed" + str(SampleSelect) + ".svg")
    #data_reduce.save_file(QXP,QYP,QYError/2,info.OutPath + '/' +"IQ_NormedD3_" + SampleName + ".dat")
    #data_reduce.save_file4(QXP,QYP,QYError/2,QXError,info.OutPath +'/' + "IQ_Normed_with_QXErrorD3_" + SampleName + ".dat")
#    data_reduce.save_image_with_error_bar(QXP,QYP,QYError/2, info.OutPath + '/' +"IQ_NormedD3" + SampleName + ".svg", label = SampleName)
    QZ2D = smooth_2D.smooth(QZ2D,0.5)

    data_reduce.save_matrix(QZ2D,FileName = info.OutPath + '/' + "QXQYMatrix_2D_" + SampleName + ".dat")
    nQX = len(QX2D)
    nQY = len(QY2D)
    QY2Dp = np.ones((nQY,nQX))*QY2D[:,None]
    QX2Dp = np.ones((nQX,nQY))*QX2D
    data_reduce.save_file4_QXY(np.ravel(QX2Dp),np.ravel(QY2Dp),np.ravel(QZ2D),np.ravel(QZError2D),info.OutPath + '/' + "QXQYI_3D" + SampleName + ".dat")
    data_reduce.save_file2(QX2D,QY2D,info.OutPath + '/' + "QXQY" + SampleName + ".dat")
    data_reduce.save_file2(data_reduce.WavelengthArray,SampleTrans,info.OutPath + '/' + 'Transmission' + '/' + "Sample_Trans" + SampleName + ".dat")
    data_reduce.save_file2(data_reduce.WavelengthArray,CellTrans,info.OutPath + '/' +'Transmission' + '/' + "Cell_Trans" + SampleName + ".dat")

    data_reduce.mat_plot_2d(QZ2D.copy(),show = info.debug, FileName = info.OutPath + '/'+ 'MatPlots' +'/' + "Mat2DPlot_logscale_" + SampleName,logscale = info.DataReduce2DScaleLog)

    #print(QZ2D[QZ2D>0])
#    data_reduce.plot_data_2d(QX2D,QY2D,QZ2D,show= info.debug, FileName = info.OutPath + '/' + "QXQY2DPlot" + SampleName)
    data_reduce.plot_data_2d(QX2D,QY2D,QZ2D.copy(),show= info.debug, FileName = info.OutPath + '/'+ 'LogPlots' +'/' + "QXQY2DPlot_logscale_" + SampleName, logscale = info.DataReduce2DScaleLog)
    data_reduce.plot_data_2d(QX2D,QY2D,QZ2D.copy(),show= info.debug, FileName = info.OutPath + '/' + 'LinearPlots' + '/'+ "QXQY2DPlot_linearscale_" + SampleName, logscale = False)

    # data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(AirDirect,axis = 0),"AirDirect_Lambda" + SampleName + ".dat")
    # data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(SampleDirect,axis = 0),"SampleDirect_Lambda" + SampleName + ".dat")
    # data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(SampleScattering,axis = 0),"SampleScattering_Lambda" + SampleName + ".dat")

       
def reduce_one_data(data_reduce, SampleName,SampleScattering,CellScattering,SampleDirect,CellDirect,AirDirect,SampleThickness,info):
    global data_reduce0
#    SampleScatteringRun = r"RUN000" + SampleScattering
#    SampleDirectRun = r"RUN000" + SampleDirect
#    CellScatteringRun = r"RUN000" + CellScattering
#    CellDirectRun = r"RUN000" + CellDirect
#    AirDirectRun = r"RUN000" + AirDirect
    SampleScatteringRun = get_run_fold(SampleScattering) #r"RUN" + str('0'*(7-len(SampleScattering))) + SampleScattering
    SampleDirectRun =  get_run_fold(SampleDirect) #r"RUN"  + str('0'*(7-len(SampleDirect))) + SampleDirect
    CellScatteringRun =  get_run_fold(CellScattering) #r"RUN"  + str('0'*(7-len(CellScattering))) + CellScattering
    CellDirectRun =  get_run_fold(CellDirect) #r"RUN"  + str('0'*(7-len(CellDirect))) + CellDirect
    AirDirectRun =  get_run_fold(AirDirect) #r"RUN"  + str('0'*(7-len(AirDirect))) + AirDirect
    data_reduce.SampleThickness = float(SampleThickness)

    ExperimentTimeSample = data_reduce.get_experimental_time_info(SampleScatteringRun)
    ExperimentTimeCell = data_reduce.get_experimental_time_info(CellScatteringRun)


#    SampleScatteringRun = r"RUN" + str('0'*(7-len(SampleScattering))) + SampleScattering
#    SampleDirectRun = r"RUN"  + str('0'*(7-len(SampleDirect))) + SampleDirect
#    CellScatteringRun = r"RUN"  + str('0'*(7-len(CellScattering))) + CellScattering
#    CellDirectRun = r"RUN"  + str('0'*(7-len(CellDirect))) + CellDirect
#    AirDirectRun = r"RUN"  + str('0'*(7-len(AirDirect))) + AirDirect

#    data_reduce.SampleThickness = float(SampleThickness)

    AirDirect,AirDirectPC = data_reduce0.load_data(AirDirectRun)
    CellDirect,CellDirectPC = data_reduce0.load_data(CellDirectRun)*info.DirectFactorCell
    SampleDirect,SampleDirectPC = data_reduce0.load_data(SampleDirectRun)*info.DirectFactorSample
    #plt.plot(np.sum(AirDirect,axis = 1))
    #plt.yscale('log')
    #plt.plot(np.sum(SampleDirect,axis = 1))
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.show()
    #plt.close()
    #AirDirect = AirDirect/AirDirectPC
    #CellDirect = CellDirect*info.DirectFactorCell/CellDirectPC
    #SampleDirect = SampleDirect*info.DirectFactorSample/SampleDirectPC
   
    CellScattering,CellPC = data_reduce.load_data(CellScatteringRun) #*0
    SampleScattering,SamplePC = data_reduce.load_data(SampleScatteringRun)
    #print('afterload: ',np.sum(SampleScattering))    
    I0X,I0Y = data_reduce.direct_beam_integrate_to_lambda(AirDirect)
    ISX,ISY = data_reduce.direct_beam_integrate_to_lambda(SampleDirect)
    ICX,ICY = data_reduce.direct_beam_integrate_to_lambda(CellDirect)
    SampleTrans = data_reduce.trans_calc(ISY*info.DirectFactorSample,I0Y)
    CellTrans = data_reduce.trans_calc(ICY*info.DirectFactorCell,I0Y)
    #SampleTrans = data_reduce.trans_calc(data_reduce.integrate_x(SampleDirect), data_reduce.integrate_x(AirDirect))
    I0YP = I0Y
    ##print(info.DirectFactorSample)
    #print(ISY/I0YP)

###################################
#   data_reduce.save_file2(I0X,I0Y,info.OutPath + '/' + "I_Air_Lambda_D3" + SampleName + ".dat")
#   data_reduce.save_file2(ISX,ISY,info.OutPath + '/' + "I_Sample_Lambda_D3" + SampleName + ".dat")
#   data_reduce.save_file2(ICX,ICY,info.OutPath + '/' + "I_Cell_Lambda_D3" + SampleName + ".dat")
#   data_reduce.data_plot_xy(I0X,I0Y,label='I_Air_Lambda plot_D3', show = False,FileName = info.OutPath + '/'+ "I_Air_Lambda_D3" + SampleName)
#   data_reduce.data_plot_xy(ISX,ISY,label='I_Sample_Lambda plot_D3', show = False,FileName = info.OutPath + '/'+ "I_Sample_Lambda_D3" + SampleName)
#   data_reduce.data_plot_xy(ICX,ICY,label='I_Cell_Lambda plot_D3', show = False ,FileName = info.OutPath + '/'+ "I_Cell_Lambda_D3" + SampleName)

#   ISSD3X,ISSD3Y = data_reduce.direct_beam_integrate_to_lambda(SampleScattering)
#   ICSD3X,ICSD3Y = data_reduce.direct_beam_integrate_to_lambda(CellScattering)

#   data_reduce.save_file2(ISSD3X,ISSD3Y,info.OutPath + '/' + "I_Sample_Scattering_Lambda_D3" + SampleName + ".dat")
#   data_reduce.save_file2(ICSD3X,ICSD3Y,info.OutPath + '/' + "I_Cell_Scattering_Lambda_D3" + SampleName + ".dat")
#   data_reduce.data_plot_xy(ISSD3X,ISSD3Y,label='I_Sample_Scattering_Lambda plot_D3', show = False,FileName = info.OutPath + '/'+ "I_Sample_Scattering_Lambda_D3" + SampleName)
#   data_reduce.data_plot_xy(ICSD3X,ICSD3Y,label='I_Cell_Scattering_Lambda plot_D3', show = False ,FileName = info.OutPath + '/'+ "I_Cell_Scattering_Lambda_D3" + SampleName)
###################################333


    #data_reduce.data_plot_xyz(I0X,SampleTrans,zero_divide(ISY/I0Y,label='Sample_Trans Fit',label2 = 'Sample_Trans', show = False,FileName = info.OutPath + '/'+ "Sample_Trans_D2" + SampleName)
    #data_reduce.data_plot_xyz(I0X,CellTrans,ICY/I0Y,label='Cell_Trans Fit', label2 = 'Cell_Trans', show = False ,FileName = info.OutPath + '/'+ "Cell_Trans_D2" + SampleName)


    #CellTrans = data_reduce.trans_calc(data_reduce.integrate_x(CellDirect), data_reduce.integrate_x(AirDirect))
    SampleTransNormed = zero_divide(SampleScattering, SampleTrans)
    CellTransNormed = zero_divide(CellScattering, CellTrans) # 
    #I0X,I0Y = data_reduce.direct_beam_integrate_to_lambda(AirDirect)
   # plt.plot(I0X,I0Y)#plot the I0
    I0 = I0Y*data_reduce.I0Scale
    I0Smoothed = data_reduce.moving_average(I0,5)
    I0Y = I0Smoothed
#    I0TofArray = np.linspace(data_reduce.TimeDelay,data_reduce.TimeDelay + data_reduce.TOF,data_reduce.WaveBins)
#    I0WavelengthArray = data_reduce.const*I0TofArray/(data_reduce.SamplePos + data_reduce.L2)
    I0X = np.hstack((I0X - data_reduce.TOF, I0X, I0X + data_reduce.TOF))
    I0Y = np.hstack((np.zeros_like(I0Y),I0Y,np.zeros_like(I0Y)))
    #f = interpolate.interp1d(I0X, I0Y,fill_value="extrapolate",bounds_error = False)
    #I0Interpolated = data_reduce.I0_interp(data_reduce.WavelengthArray2,f)
    I0Interpolated = np.interp(data_reduce.WavelengthArray2,I0X,I0Y,left = 0, right = 0) 

    if info.debug is True and info.debug2 is True:
        SampleSumed = np.sum(np.sum(SampleTransNormed[45:55],axis = 0))
        Sample50Sumed = np.sum(np.sum(SampleTransNormed[0:30],axis = 0))
        Sample99Sumed = np.sum(np.sum(SampleTransNormed[89:99],axis = 0))
        I0Sumed = np.sum(I0Interpolated[50]/10)


        plt.plot(data_reduce.WavelengthArray2[50],I0Interpolated[50]/10,label = 'I(Lambda)AirDirect')
        plt.plot(data_reduce.WavelengthArray2[50],I0Sumed/SampleSumed*np.sum(SampleTransNormed[45:55],axis = 0),label = 'I(Lambda)SampleScatteringDmiddle')
        plt.plot(data_reduce.WavelengthArray2[0],I0Sumed/Sample50Sumed*np.sum(SampleTransNormed[0:30],axis = 0),label = 'I(Lambda)SampleScatteringD3_Rmin')
        plt.plot(data_reduce.WavelengthArray2[99],I0Sumed/Sample99Sumed*np.sum(SampleTransNormed[89:99],axis = 0),label = 'I(Lambda)SampleScatteringD3_Rmax')

        plt.title(SampleName)
        #plt.xlim(1,15)
        plt.legend(fancybox=True, framealpha=0.01,frameon = False)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Counts (n/s/Å)')
        plt.yscale('log')
        plt.show()
        plt.close()
     #
        #print('I_SampleDirect_D3:',np.sum(data_reduce.load_data(SampleDirectRun)))
        #print('I_SampleScattering_D3:',np.sum(data_reduce.load_data(SampleScatteringRun)))
    #print('aftertransnormed',np.sum(SampleTransNormed)) 
    
    QXP,QYP,QYError,QXError,QYSample,QYSampleError,QYCell,QYCellError = data_reduce.translate_to_q(I0Interpolated,AirDirectPC,data_reduce.I0Scale,SampleTransNormed,SamplePC,SampleTrans,CellTransNormed,CellPC,CellTrans)
    if info.ExtraBkg > 0:
        QYCell = QYCell - info.ExtraBkg    
    ###data_reduce.save_image(QXP,QYP,"IQ_Normed" + str(SampleSelect) + ".svg")
    #data_reduce.save_file(QXP,QYP,QYError,info.OutPath + '/' +"IQ_NormedD3_" + SampleName + ".dat")
    data_reduce.save_file4(QXP,QYP,QYError,QXError,info.OutPath +'/' + "IQ_Normed_with_QXErrorD3_OnlySample_" + SampleName + ".dat",ExperimentTimeSample)
    data_reduce.save_file4(QXP,QYSample,QYSampleError,QXError,info.OutPath +'/' + "IQ_Normed_with_QXErrorD3_Sample+Cell_" + SampleName + ".dat",ExperimentTimeSample)
    data_reduce.save_file4(QXP,QYCell,QYCellError,QXError,info.OutPath +'/' + "IQ_Normed_with_QXErrorD3_OnlyCell_" + SampleName + ".dat",ExperimentTimeCell)

#    data_reduce.save_image_with_error_bar(QXP,QYP,QYError/2, info.OutPath + '/' +"IQ_NormedD3" + SampleName + ".svg", label = SampleName)
#    print(info.DirectFactorSample)
#    print(zero_divide(ISY*info.DirectFactorSample,I0YP))
   
#    data_reduce.save_file1(data_reduce.XArray,info.OutPath + '/' + "XArray" + SampleName + ".dat")
#    data_reduce.save_file1(data_reduce.YArray,info.OutPath + '/' + "YArray" + SampleName + ".dat")  
#
    data_reduce.save_file3(data_reduce.WavelengthArray,SampleTrans,zero_divide(ISY*info.DirectFactorSample,I0YP),info.OutPath + '/' + 'Transmission' + '/' + "Sample_Trans" + SampleName + ".dat")
    data_reduce.save_file3(data_reduce.WavelengthArray,CellTrans,zero_divide(ICY*info.DirectFactorSample,I0YP),info.OutPath + '/' +'Transmission' + '/' + "Cell_Trans" + SampleName + ".dat")

    data_reduce.data_plot_xyz(data_reduce.WavelengthArray,SampleTrans,zero_divide(ISY*info.DirectFactorSample,I0YP),label='Sample_Trans Fit',label2 = 'Sample_Trans', show = False,FileName = info.OutPath + '/'+'Transmission' + '/' + SampleName+ "_SampleTrans")
    data_reduce.data_plot_xyz(data_reduce.WavelengthArray,CellTrans,zero_divide(ICY*info.DirectFactorSample,I0YP),label='Cell_Trans Fit', label2 = 'Cell_Trans', show = False ,FileName = info.OutPath + '/'+'Transmission' + '/' + SampleName + '_CellTrans')
    # data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(AirDirect,axis = 0),"AirDirect_Lambda" + SampleName + ".dat")
    # data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(SampleDirect,axis = 0),"SampleDirect_Lambda" + SampleName + ".dat")
    # data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(SampleScattering,axis = 0),"SampleScattering_Lambda" + SampleName + ".dat")
#    return AirDirect,AirDirectPC,SampleDirect,SampleDirectPC,CellDirect,CellDirectPC


def load_air_direct(RunNum,info):
    global data_reduce
    DataFileName = info.DataFold + "/" + str('RUN'+ str('0'*(7-len(RunNum))) + RunNum) + "/" + "detector.nxs"
    f = h5py.File(DataFileName, "r")
    data1 = f["/csns/instrument/module32/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module31/histogram_data"][()] #got the right bank data
    tmp = info.WaveBins
    try:
        freq_ratio = f["/csns/Freq_ratio"][()]
    except:
        freq_ratio = 1

    if int(RunNum[-5:]) >= 4102:
        TofPoints = 500*freq_ratio
    else:
        TofPoints = 5000
    tmp2 = int(TofPoints/tmp)

    #tmp2 = int(5000/tmp)
    Data1Reshaped = np.reshape(data1,(64,250,tmp,tmp2))
    Data1Reshaped2 = np.sum(Data1Reshaped,axis = 3)
    Data2Reshaped = np.reshape(data2,(64,250,tmp,tmp2))
    Data2Reshaped2 = np.sum(Data2Reshaped,axis = 3)
    DataStacked = np.vstack((Data2Reshaped2,Data1Reshaped2))
    FallingPixels = falling_distance(data_reduce.WavelengthArray,data_reduce.L1Direct,data_reduce.L2)/data_reduce.TubeHeight
    DataStacked = falling_correction(DataStacked,FallingPixels)
    return DataStacked         

def guassian(x,A,sigma,mu):
    return A*np.exp(-(x-mu)**2/sigma**2)   

def guassian_fit(xx,yy):
    import scipy.optimize
    '''对一维数据xx，yy进行高斯峰拟合，返回峰强A,峰宽sigma和峰位mu'''
    sigma_guess = (xx[-1]-xx[0])
    mu_guess = np.average(xx)
    aa,bb = scipy.optimize.curve_fit(lambda x,A,sigma,mu: guassian(x,A,sigma,mu),xx,yy,p0=[3000,sigma_guess,mu_guess])
    return aa[0],aa[1],aa[2]

def find_peaks(arr,peak_width):
    '''根据输入的峰的宽度peak_width，找到输入的一列数据arr的峰位所在的点，返回峰位点的列表'''
    sum_match = []
    peak_pos = []
    peak_pos2 = []
    for match in np.arange(len(arr)-peak_width):
        upper = match+peak_width
        summ = np.sum(arr[match:upper])
        sum_match.append((match,summ))
    for item in sum_match[peak_width:-peak_width]:
        if sum_match[item[0]][1] > sum_match[item[0]-1][1] and sum_match[item[0]][1] > sum_match[item[0]+1][1]:
            up_bound = item[0]+peak_width
            matched_pos = np.array(sum_match[item[0]:up_bound])
            ave_pos = np.average(matched_pos[:,0])
            peak_pos2.append((ave_pos,item[1]))
            #print(sum_match[item[0]:up_bound])
            peak_pos.append((item[0]+int(peak_width/2),item[1]))
    return peak_pos2

def get_big_peaks(arr,peak_pos,num,peak_width):
    '''在一列数据arr中，已知所有的峰的粗略位置peak_pos[0]和峰强peak_pos[1]，和设定的peak_width，
    利用高斯拟合找到前num个最强峰的精确位置，返回最强峰位置和峰高的二维数组'''
    peak_height_sorted = sorted(peak_pos,key = lambda x: x[1], reverse = True)
    big_peaks_sorted = sorted(peak_height_sorted[:num],key = lambda x: x[0])
    mu_array = []
    #print(big_peaks_sorted,'--------')
    big_peaks_array = np.array(big_peaks_sorted)
    #print(big_peaks_array,'--------')
    for i,peaks in enumerate(big_peaks_array[:,0]):
        low_bound,high_bound = int(peaks-peak_width/2),int(peaks+peak_width/2)
        xx = np.arange(low_bound,high_bound)
        yy = arr[low_bound:high_bound]     
#        plt.plot(xx,yy)
        A,sigma,mu = guassian_fit(xx,yy)
#        plt.plot(xx,guassian(xx,A,sigma,mu))
#        plt.show()
        big_peaks_array[i,0] = mu 
        mu_array.append(mu)
    out3 = np.ones(big_peaks_array.shape)
    out3[:,0] = mu_array
    out3[:,1] = big_peaks_array[:,1]
    return out3    

def get_center(RunNum,info):
    global data_reduce
    DataFold = info.DataFold
    #RunInfoFold = info.RunInfoFold

    data_reduce = data_reduce(DataFold, info)

    data = load_air_direct(RunNum,data_reduce)
    #StartWave = data_reduce.StartWave + 50 #50
    #StopWave = StartWave + 10 # 10 #data_reduce.StopWave
    DataCut = np.sum(data,axis = 2)
    YIntegrated = np.sum(DataCut,axis = 1)
    XIntegrated = np.sum(DataCut,axis = 0)
    if data_reduce.A2Small <= 1.5:
        PeakWidth = 5
    else:
        PeakWidth = 5 
    YPeaks = find_peaks(XIntegrated,PeakWidth)
    YCenter = get_big_peaks(XIntegrated,YPeaks,1,PeakWidth)[0][0]
    XPeaks = find_peaks(YIntegrated,PeakWidth)
    XCenter = get_big_peaks(YIntegrated,XPeaks,1,PeakWidth)[0][0]#-24/8.5
    print('X center is :' + str(round(XCenter,2)) + 'th Pixels  Y center is :' + str(round(YCenter)) + 'th Pixels')
    #print(np.average(data_reduce.WavelengthArray),data_reduce.L1, data_reduce.L2)
    #FallingDistance = data_reduce.falling_distance(data_reduce.WavelengthArray[StartWave:StopWave], data_reduce.L1Direct, data_reduce.L2)
    
#    FallingDistance = falling_distance(self.WavelengthArray,self.L1,self.L2)
    #YCenter = YCenter - np.average(FallingDistance)/4 #,weights = np.sum(np.sum(data,axis = 0),axis = 0)[StartWave:StopWave])
    #print('Gravity corrected X center is :' + str(XCenter) + 'Y center is :' + str(YCenter))
    return XCenter,YCenter

