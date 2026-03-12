# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 06:27:53 2023

@author: zuots
"""
import sys
import os
import re
import glob
import math
import datetime
from datetime import datetime

import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
from scipy import integrate, interpolate
from numpy.polynomial import polynomial as P
import pickle
from matplotlib import pyplot as plt
from numba import jit
import importlib as imp
import h5py

# Import new modules
from input_module import InputModule
from output_module import OutputModule
from calculation_module import CalculationModule
from gravity_correction import falling_correction
from gravity_correction import falling_distance

# Import local modules
import data_reduce_D3
import D3
import efficiency_calc

global data_reduce
global data_reduce0
global D3
global test


class data_reduce:
    def __init__(self, DataFold, InstrumentInfo):
        # Basic configuration
        self.DataFold = DataFold
        self.info = InstrumentInfo
        self.const = 3956.2
        self.formated_time = self.get_now()
        
        # Instrument information
        self.MaskSwitch = self.info.MaskSwitch
        self.WaveBinsSelected = self.info.WaveBinsSelectedD1
        self.IncludeD1 = self.info.IncludeD1
        self.TimeFixFactor = self.info.TimeFixFactor
        self.StartWave = self.info.StartWave
        self.StopWave = self.info.StopWave
        self.CellScatteringFactor = self.info.CellScatteringFactor
        self.SampleScatteringFactor = self.info.SampleScatteringFactor
        
        # Instrument parameters
        self.L1 = self.info.L1 + self.info.SampleDisplace  # 72.5 mm
        self.L2_13 = self.info.D1_L2 * self.info.L2_factor - self.info.SampleDisplace  # mm
        self.L2_24 = self.info.D1_L2 * self.info.L2_factor + 430 - self.info.SampleDisplace  # mm
        self.L2 = (self.L2_13 + self.L2_24) / 2
        self.SamplePos = self.info.SamplePos + self.info.SampleDisplace  # mm
        self.A1 = self.info.A1  # mm
        self.A2 = self.info.A2  # mm
        self.A2Small = self.info.A2Small  # mm
        self.TimeDelay = self.info.TimeDelayD1  # 27.96 ms (6埃, 2.2埃和4埃的延时分别为: 52.7ms, 19.32ms和35.135ms)
        self.TimeDelayD3 = self.info.TimeDelayD3  # 49.85 ms
        self.TOF = self.info.TOF  # 40 ms
        self.I0Scale = self.info.IDirectBeamScale
        
        # Data parameters
        self.QMin = self.info.QMin  # 0.001 A^-1
        self.QMax = self.info.QMax  # 2 A^-1
        self.QBins = self.info.QBins  # 120
        self.WaveMin = self.info.WaveMin  # A
        self.WaveMax = self.info.WaveMax  # A
        self.TofMin13 = self.TimeFixFactor + (self.SamplePos + self.L2_13) * self.WaveMin / self.const
        self.TofMax13 = self.TimeFixFactor + (self.SamplePos + self.L2_13) * self.WaveMax / self.const
        self.TofMin24 = self.TimeFixFactor + (self.SamplePos + self.L2_24) * self.WaveMin / self.const
        self.TofMax24 = self.TimeFixFactor + (self.SamplePos + self.L2_24) * self.WaveMax / self.const
        self.TofBins = 250
        self.WaveBins = self.TofBins
        self.XBins = 128
        self.YBins = 250
        
        # Detector parameters
        self.ModToDetector = self.SamplePos + (self.L2_13 + self.L2_24) / 2  # mm
        self.DetFactor = self.info.DetFactor  # 1.0
        # self.BankWidth = 1094 * self.DetFactor  # mm
        # self.BankHeight = 1000 * self.DetFactor  # mm
        self.PixelHeight = 4 * self.DetFactor  # mm
        self.TubeWidth = 8.5  # mm (original: ((15*0.5+8*15)*3+(2.5+8)*2)/47*self.DetFactor = 8.5851mm)
        self.TubeHeight = self.PixelHeight
        self.Tubes = 48
        self.ShortPixels = 150
        self.LongPixels = 250
        self.RMin =300      #mm
        self.RMax = 860     #mm
        self.RBins = 100          
        self.RCut = self.info.D1RCut
        self.RBinsCut = int(self.RBins - self.RBins*self.RCut/self.RMax)

        self.ModuleGap = 1.5  #mm
        self.ModuleWidth = self.TubeWidth*16
        self.BankWidth = (self.TubeWidth*48 + self.ModuleGap*2)*self.DetFactor
        self.BankHeight13 = 1000*self.DetFactor - self.TubeHeight
        self.BankHeight24 = 600*self.DetFactor - self.TubeHeight
        self.ModuleArray = np.arange(0,self.TubeWidth*16,self.TubeWidth)
        self.FirstArray = self.ModuleArray-self.BankWidth/2+self.TubeWidth/2
        self.ArrayDistance = self.ModuleWidth + self.ModuleGap
        self.X13 = np.concatenate((self.FirstArray,self.FirstArray+self.ArrayDistance,self.FirstArray+self.ArrayDistance*2))
        self.Y13 = np.arange(self.LongPixels*self.PixelHeight*(-1)/2+self.TubeHeight/2,self.LongPixels*self.PixelHeight/2+self.TubeHeight/2,+self.TubeHeight)
        self.Y24 = np.concatenate((self.FirstArray,self.FirstArray+self.ArrayDistance,self.FirstArray+self.ArrayDistance*2))
        self.X24 = np.arange(self.ShortPixels*self.PixelHeight*(-1)/2+self.TubeHeight/2,self.ShortPixels*self.PixelHeight/2+self.TubeHeight/2,+self.TubeHeight)
#        self.X13 = np.linspace(self.Tubes*self.TubeWidth*(-1)/2,self.Tubes*self.TubeWidth/2,self.Tubes)                   
#        self.Y13 = np.linspace(self.LongPixels*self.PixelHeight*(-1)/2,self.LongPixels*self.PixelHeight/2,self.LongPixels)
#        self.X24 = np.linspace(self.ShortPixels*self.PixelHeight*(-1)/2,self.ShortPixels*self.PixelHeight/2,self.ShortPixels)
#        self.Y24 = np.linspace(self.Tubes*self.TubeWidth*(-1)/2,self.Tubes*self.TubeWidth/2,self.Tubes)


#        self.CosAlpha13 = self.L2_13/np.sqrt(self.Y13**2 + self.L2_13**2)
#        self.CosAlpha24 = self.L2_24/np.sqrt(self.X24**2 + self.L2_24**2)

#        self.OriginD21 = [(-300-self.TubeWidth*self.Tubes),-500]
#        self.OriginD22 = [-300,300-3]
#        self.OriginD23 = [300-13,-500]
#        self.OriginD24 = [-300,-300-self.TubeWidth*self.Tubes]
        self.OriginD21 = [(-304-self.BankWidth),-500]
        self.OriginD22 = [-304,304]
        self.OriginD23 = [304,-500]
        self.OriginD24 = [-304,-304-self.BankWidth]        

        self.X1 = self.OriginD21[0] + self.X13 + self.BankWidth/2     #np.linspace(0,self.Tubes*self.TubeWidth,self.Tubes)
        self.X3 = self.OriginD23[0] + self.X13 + self.BankWidth/2     # np.linspace(0,self.Tubes*self.TubeWidth,self.Tubes)
        self.Y1 = self.OriginD21[1] + self.Y13 + self.LongPixels*self.PixelHeight/2   #np.linspace(0,self.LongPixels*self.PixelHeight,self.LongPixels)
        self.Y3 = self.OriginD23[1] + self.Y13 + self.LongPixels*self.PixelHeight/2   #np.linspace(0,self.LongPixels*self.PixelHeight,self.LongPixels)
        self.X2 = self.OriginD22[0] + self.X24 + self.ShortPixels*self.PixelHeight/2  #np.linspace(0,self.ShortPixels*self.PixelHeight,self.ShortPixels)
        self.X4 = self.OriginD24[0] + self.X24 + self.ShortPixels*self.PixelHeight/2  #np.linspace(0,self.ShortPixels*self.PixelHeight,self.ShortPixels)
        self.Y2 = self.OriginD22[1] + self.Y24 + self.BankWidth/2    #np.linspace(0,self.Tubes*self.TubeWidth,self.Tubes)
        self.Y4 = self.OriginD24[1] + self.Y24 + self.BankWidth/2    #np.linspace(0,self.Tubes*self.TubeWidth,self.Tubes)


        self.CosAlpha13 = self.L2_13/np.sqrt(self.Y1**2 + self.L2_13**2)
        self.CosAlpha24 = self.L2_24/np.sqrt(self.X2**2 + self.L2_24**2)
        self.CosBeta13 = self.L2_13/np.sqrt(self.X1**2 + self.L2_13**2)
        self.CosBeta24 = self.L2_13/np.sqrt(self.Y2**2 + self.L2_13**2)
        
        #####detector parameters##########      
        
        #####Other parameters##########  
        self.SampleThickness = 1   #mm
        self.DeltaLambdaRatio = 0.019*self.const/self.ModToDetector
        #self.QX = np.logspace(log10(self.QMin),log10(self.QMax),self.QBins)
        #self.QY = np.logspace(log10(self.QMin),log10(self.QMax),self.QBins)
        self.RArrayEdges = np.logspace(np.log10(self.RMin), np.log10(self.RMax), self.RBins + 1)
        self.RArray = np.sqrt(self.RArrayEdges[:-1] * self.RArrayEdges[1:])  # bin centers (geometric mean)
        self.L2_13Array = np.sqrt(self.L2_13**2 + self.RArray**2)
        self.L2_24Array = np.sqrt(self.L2_24**2 + self.RArray**2)
        self.L_13 = self.SamplePos + self.L2_13
        self.L_24 = self.SamplePos + self.L2_24
        self.L_13Array = self.SamplePos + self.L2_13Array
        self.L_24Array = self.SamplePos + self.L2_24Array
        #
        # self.WavelengthArray13 = np.linspace(self.const*self.TimeDelay/self.L_13,self.const*(self.TimeDelay + self.TOF)/self.L_13,self.WaveBins)
        # self.WavelengthArray24 = np.linspace(self.const*self.TimeDelay/self.L_24,self.const*(self.TimeDelay + self.TOF)/self.L_24,self.WaveBins)
        # self.TofArray13 = (self.SamplePos + self.L2_13Array[:,None])*self.WavelengthArray13/self.const
        # self.TofArray24 = (self.SamplePos + self.L2_24Array[:,None])*self.WavelengthArray24/self.const
        # self.WavelengthArray13 = self.const*self.TofArray13/self.L_13
        # self.WavelengthArray24 = self.const*self.TofArray24/self.L_13
        
        self.TofArray13 = np.linspace(self.TimeDelay,self.TimeDelay + self.TOF,self.WaveBins) + self.TOF/self.WaveBins/2
        self.TofArray24 = np.linspace(self.TimeDelay,self.TimeDelay + self.TOF,self.WaveBins) + self.TOF/self.WaveBins/2
        self.WavelengthArray13 = self.const*self.TofArray13/self.L_13Array[:,None]
        self.WavelengthArray24 = self.const*self.TofArray24/self.L_24Array[:,None]

        #self.TofMin13 = self.L_13Array[:,None]*self.StartWave/self.const  #可注释    注释后表示截掉氢的非弹散射部分
        #self.TofMax13 = self.L_13Array[:,None]*self.StopWave/self.const   #可注释   注释后表示截掉氢的非弹散射部分
        self.WavelengthArrayBool13 = (self.TofArray13 > self.TofMin13)*(self.TofArray13 < self.TofMax13)

        #self.TofMin24 = self.L_24Array[:,None]*self.StartWave/self.const #可注释   注释后表示截掉氢的非弹散射部分
        #self.TofMax24 = self.L_24Array[:,None]*self.StopWave/self.const  #可注释   注释后表示截掉氢的非弹散射部分
        self.WavelengthArrayBool24 = (self.TofArray24 > self.TofMin24)*(self.TofArray24 < self.TofMax24)

        
        self.WavelengthArray13_1D = np.linspace(self.WaveMin,self.WaveMax,self.WaveBins) + (self.WaveMax - self.WaveMin)/self.WaveBins/2
        self.WavelengthArray24_1D = np.linspace(self.WaveMin,self.WaveMax,self.WaveBins) + (self.WaveMax - self.WaveMin)/self.WaveBins/2
        # self.TofArray13 = self.L2_13Array[:,None]*self.WavelengthArray13/self.const
        # self.TofArray24 = self.L2_24Array[:,None]*self.WavelengthArray24/self.const
        #self.Q = np.linspace(self.QMin,self.QMax,self.QBins)
        #self.Q = np.logspace(log10(self.QMin),log10(self.QMax),self.QBins)
        self.Q = self.info.QX
        self.ThetaArray13 = np.arctan(self.RArray/self.L2_13)
        self.ThetaArray24 = np.arctan(self.RArray/self.L2_24)
        self.WavelengthArray = np.linspace(self.WaveMin,self.WaveMax,self.WaveBins)
        self.QArray13 = 4*np.pi*np.sin(self.ThetaArray13[:,None]/2)/self.WavelengthArray13
        self.QArray24 = 4*np.pi*np.sin(self.ThetaArray24[:,None]/2)/self.WavelengthArray24
        #self.mask = self.load_mask()

    def load_mask(self):
        return InputModule.load_mask('D1')


    def get_proton_charge(self, file):
        return InputModule.get_proton_charge(file)

    def get_now(self):
        return InputModule.get_now()

    def get_detector_coordinate(self):
#       D21X = np.linspace(0,self.TubeWidth*self.Tubes,self.Tubes)
#       D21Y = np.linspace(0,self.PixelHeight*self.LongPixels,self.LongPixels)
#       D22X = np.linspace(0,self.PixelHeight*self.ShortPixels,self.ShortPixels)
#       D22Y = np.linspace(0,self.TubeWidth*self.Tubes,self.Tubes)
#       D23X = np.linspace(0,self.TubeWidth*self.Tubes,self.Tubes)
#       D23Y = np.linspace(0,self.PixelHeight*self.LongPixels,self.LongPixels)
#       D24X = np.linspace(0,self.PixelHeight*self.ShortPixels,self.ShortPixels)
#       D24Y = np.linspace(0,self.TubeWidth*self.Tubes,self.Tubes)        
        D21XP = self.X1 #D21X + self.OriginD21[0]
        D21YP = self.Y1 #D21Y + self.OriginD21[1]
        D22XP = self.X2 #D22X + self.OriginD22[0]
        D22YP = self.Y2 #D22Y + self.OriginD22[1]
        D23XP = self.X3 #D23X + self.OriginD23[0]
        D23YP = self.Y3 #D23Y + self.OriginD23[1]
        D24XP = self.X4 #D24X + self.OriginD24[0]
        D24YP = self.Y4 #D24Y + self.OriginD24[1]
        D2X = {0:D21XP,1:D22XP,2:D23XP,3:D24XP}
        D2Y = {0:D21YP,1:D22YP,2:D23YP,3:D24YP}
        return D2X,D2Y

    def detector_group(self):
        # Check if saved GroupX and GroupY files exist
        filename_x = f'./npyfiles/D1GroupX_{self.WaveMin}-{self.WaveMax}A.pkl'
        filename_y = f'./npyfiles/D1GroupY_{self.WaveMin}-{self.WaveMax}A.pkl'
        load_successful = False
        
        if os.path.exists(filename_x) and os.path.exists(filename_y):
            try:
                # Load the saved GroupX and GroupY
                with open(filename_x, 'rb') as f:
                    GroupX = pickle.load(f)
                with open(filename_y, 'rb') as f:
                    GroupY = pickle.load(f)
                load_successful = True
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                print(f"Error loading D1 pickle files: {e}. Regenerating GroupX and GroupY...")
                # Delete the corrupted files
                try:
                    os.remove(filename_x)
                    os.remove(filename_y)
                except:
                    pass
        
        if not load_successful:
            # Generate GroupX and GroupY using the static method
            GroupX, GroupY = CalculationModule.detector_group_d1d2(self)
        
        return GroupX, GroupY

    def grouping(self,D2Data):        
        return CalculationModule.grouping_d1d2(self, D2Data)

    def grouping_mask0(self,D2Data,LowerRatio,HigherRatio,WavePeakMin,WavePeakMax,LambdaSinTheta_2Min):        
        GroupX,GroupY = self.detector_group()
        mask = {0:np.ones_like(D2Data[0]),1:np.ones_like(D2Data[1]),2:np.ones_like(D2Data[2]),3:np.ones_like(D2Data[3])}       
        tmp1 = (self.WavelengthArray13 < WavePeakMin)+(self.WavelengthArray13 > WavePeakMax)
        tmp2 = (self.WavelengthArray24 < WavePeakMin)+(self.WavelengthArray24 > WavePeakMax)
        Judge3 = {0:tmp1,1:tmp2,2:tmp1,3:tmp2}  
        tmp3 = self.WavelengthArray13*np.sin(self.ThetaArray13[:,None]/2) > LambdaSinTheta_2Min
        tmp4= self.WavelengthArray24*np.sin(self.ThetaArray24[:,None]/2) > LambdaSinTheta_2Min
        Judge4 = {0:tmp3,1:tmp4,2:tmp3,3:tmp4} 
        for t in range(len(D2Data)):
            for i in np.arange(self.RBins):
                tmp = np.average(D2Data[t][GroupY[t][i],GroupX[t][i]],axis = 0)
                Judge1 = D2Data[t][GroupY[t][i],GroupX[t][i]] > tmp*LowerRatio
                Judge2 = D2Data[t][GroupY[t][i],GroupX[t][i]] < tmp*HigherRatio
                mask[t][GroupY[t][i],GroupX[t][i]] = mask[t][GroupY[t][i],GroupX[t][i]]*Judge1*Judge2*Judge3[t][i]*Judge4[t][i]
        return mask

    def grouping_mask(self,D2Data,XMinD2,XMaxD2,YMinD2,YMaxD2):
        return CalculationModule.grouping_mask_d1d2(self, D2Data, XMinD2, XMaxD2, YMinD2, YMaxD2)


    def azimuthal_mask(self,D2Data,PhiMin,PhiMax):
        return CalculationModule.azimuthal_mask_d1d2(self, D2Data, PhiMin, PhiMax)

    def solid_angle0(self):
        GroupX,GroupY = self.detector_group()
        PixelArea = self.TubeWidth*self.PixelHeight
        SolidR = np.zeros(len(GroupX))[:,None]*np.zeros(self.RBins)
        for t in range(len(GroupX)):
            for i in range(len(GroupX[t])):
                if t == 0 or t == 2:
                    SolidR[t][i] = PixelArea*len(GroupX[t][i])/(self.L2_13**2 + self.RArray[i]**2)*(self.L2_13/np.sqrt(self.RArray[i]**2+self.L2_13**2))**3
                if t == 1 or t == 3:
                    SolidR[t][i] = PixelArea*len(GroupX[t][i])/(self.L2_24**2 + self.RArray[i]**2)*(self.L2_24/np.sqrt(self.RArray[i]**2+self.L2_24**2))**3
        return SolidR    

    def solid_angle(self):
        return CalculationModule.solid_angle_d1d2(self)

    def solid_angle_2d(self):
        return CalculationModule.solid_angle_2d_d1d2(self)

    def efficiency_matrix(self):    
        calc1 = efficiency_calc.efficiency(self.L2_13,self.X1,self.Y1,self.WavelengthArray13_1D)
        calc2 = efficiency_calc.efficiency(self.L2_24,self.Y2,self.X2,self.WavelengthArray24_1D)
        calc3 = efficiency_calc.efficiency(self.L2_13,self.X3,self.Y3,self.WavelengthArray13_1D)
        calc4 = efficiency_calc.efficiency(self.L2_24,self.Y4,self.X4,self.WavelengthArray24_1D)      
        effic1 = calc1.pixel_efficiency()
        effic2 = calc2.pixel_efficiency()
        effic3 = calc3.pixel_efficiency()
        effic4 = calc4.pixel_efficiency()
        np.save(r'./npyfiles/D1_effic1_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',effic1)
        np.save(r'./npyfiles/D1_effic2_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',effic2)       
        np.save(r'./npyfiles/D1_effic3_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',effic3)
        np.save(r'./npyfiles/D1_effic4_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',effic4) 
        return effic1,effic2,effic3,effic4  

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
        # 确保RunNum是正确的格式
        if isinstance(RunNum, str) and RunNum.startswith("RUN"):
            run_dir = RunNum
        else:
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


    def load_data_origin(self,RunNum):
        DataFileName = self.DataFold + "/" + str(RunNum) + "/" + "detector.nxs"    
        #RunInfoFileName = glob.glob(self.RunInfoFold + '/' + str(RunNum) + '/' + '**.xml')[0]
        f = h5py.File(DataFileName, "r")
        data1 = f["/csns/instrument/module11/histogram_data"][()] #got the left bank data
        data2 = f["/csns/instrument/module12/histogram_data"][()] #got the top bank data
        data3 = f["/csns/instrument/module13/histogram_data"][()] #got the right bank data
        data4 = f["/csns/instrument/module14/histogram_data"][()] #got the bottom bank data
        ProtonCharge = f["/csns/proton_charge"][()]
        #reshape the data to arrays
        tmp = self.WaveBins
        try:
            freq_ratio = f["/csns/Freq_ratio"][()]
        except:
            freq_ratio = 1

        if int(RunNum[-6:]) >= 4102:
            TofPoints = 500*freq_ratio
        else:
            TofPoints = 5000
        tmp = int(TofPoints/tmp)

        #tmp = int(5000/self.TofBins)
        Data1Reshaped = np.sum(np.reshape(data1,(self.Tubes,self.LongPixels,self.TofBins,tmp)),axis = 3)#/self.CosAlpha13[:,None]
        Data2Reshaped = np.sum(np.reshape(data2,(self.Tubes,self.ShortPixels,self.TofBins,tmp)),axis = 3)#/self.CosAlpha24[:,None]
        Data3Reshaped = np.sum(np.reshape(data3,(self.Tubes,self.LongPixels,self.TofBins,tmp)),axis = 3)#/self.CosAlpha13[:,None]
        Data4Reshaped = np.sum(np.reshape(data4,(self.Tubes,self.ShortPixels,self.TofBins,tmp)),axis = 3)#/self.CosAlpha24[:,None]

        try:
            effic1 = np.load(r'./npyfiles/D1_effic1_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',allow_pickle=True)
            effic2 = np.load(r'./npyfiles/D1_effic2_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',allow_pickle=True)
            effic3 = np.load(r'./npyfiles/D1_effic3_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',allow_pickle=True)
            effic4 = np.load(r'./npyfiles/D1_effic4_' + str(self.WaveMin) + '-' + str(self.WaveMax) + '.npy',allow_pickle=True)
        except:
            effic1,effic2,effic3,effic4 = self.efficiency_matrix() 

       #effic1,effic2,effic3,effic4 = self.efficiency_matrix()
        Data1Reshaped = Data1Reshaped/effic1
        Data2Reshaped = Data2Reshaped/effic2
        Data3Reshaped = Data3Reshaped/effic3
        Data4Reshaped = Data4Reshaped/effic4

        FallingPixels13 = falling_distance(self.WavelengthArray,self.L1,self.L2_13)/self.TubeHeight
        FallingPixels24 = falling_distance(self.WavelengthArray,self.L1,self.L2_24)/self.TubeWidth
        Data1Reshaped = falling_correction(Data1Reshaped,FallingPixels13)
        Data3Reshaped = falling_correction(Data3Reshaped,FallingPixels13)
        Data2Reshaped = falling_correction(np.transpose(Data2Reshaped,(1,0,2)),FallingPixels24)
        Data4Reshaped = falling_correction(np.transpose(Data4Reshaped,(1,0,2)),FallingPixels24)


        Data1Reshaped = np.transpose(Data1Reshaped,(1,0,2))
        Data3Reshaped = np.transpose(Data3Reshaped,(1,0,2))
        Data2Reshaped = np.transpose(Data2Reshaped,(1,0,2))
        Data4Reshaped = np.transpose(Data4Reshaped,(1,0,2))

        
#        if int(RunNum[-4:]) >= 4608 and int(RunNum[-4:]) <= 4617:
#            for i in range(1,5):
#                DataStacked = locals()['Data'+str(i)+'Reshaped']
#                DataTmp1= DataStacked[:,:,138:250].copy()
#                DataTmp2 = DataStacked[:,:,0:138].copy()
#                DataStacked[:,:,0:112] = DataTmp1
#                DataStacked[:,:,112:250] = DataTmp2
#        if int(RunNum[-4:]) >= 8837 and int(RunNum[-4:]) <= 8860:
#            for i in range(1,5):
#                DataStacked = locals()['Data'+str(i)+'Reshaped']
#                DataTmp1= DataStacked[:,:,110:250].copy()
#                DataTmp2 = DataStacked[:,:,0:110].copy()
#                DataStacked[:,:,0:140] = DataTmp1
#                DataStacked[:,:,140:250] = DataTmp2


        if self.MaskSwitch == True:
            mask = [] #np.load('masks/D1Mask' + str(t) + '.npy') #mask = self.mask
            for i in range(4):
                tmp = np.load('masks/D1Mask' + str(i) + '.npy',allow_pickle=True)
                mask.append(tmp)
#                ratio = (np.sum(tmp)/np.sum(np.ones_like(tmp)))
            D1 = {0:Data1Reshaped*mask[0][:,:,None],1:Data2Reshaped*mask[1][:,:,None],2:Data3Reshaped*mask[2][:,:,None],3:Data4Reshaped*mask[3][:,:,None]}
        else:
            D1 = {0:Data1Reshaped,1:Data2Reshaped,2:Data3Reshaped,3:Data4Reshaped}
        return D1,ProtonCharge

    def load_data(self,RunNum):
        D2,ProtonCharge = self.load_data_origin(RunNum)
        #R grouping of the data
        #print(np.sum(D2[0]))
        #print(np.sum(D2[1]))
        #BeforeGrouping = np.sum(D2[0])+np.sum(D2[1])+np.sum(D2[2])+np.sum(D2[3])
        DataGrouped = self.grouping(D2)
        DataGrouped[0] = DataGrouped[0]*self.WavelengthArrayBool13
        DataGrouped[1] = DataGrouped[1]*self.WavelengthArrayBool24
        DataGrouped[2] = DataGrouped[2]*self.WavelengthArrayBool13
        DataGrouped[3] = DataGrouped[3]*self.WavelengthArrayBool24
        #print(np.sum(DataGrouped))
        #AfterGrouping = np.sum(DataGrouped)
        DataNormed = DataGrouped #/ProtonCharge*9000*390701        
        return DataNormed,ProtonCharge

    def direct_beam_integrate_to_lambda(self, xp, data):
        # data = self.load_data(RunNum)
        xx13 = self.WavelengthArray13
        xx24 = self.WavelengthArray24
        # xp = self.WavelengthArray
        yp = np.zeros_like(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                if i == 0 or i == 2:
                    f = interpolate.interp1d(xx13[j], data[i][j], fill_value="extrapolate", bounds_error=False)
                    yp[i][j] = f(xp)
                elif i == 1 or i == 3:
                    f = interpolate.interp1d(xx24[j], data[i][j], fill_value="extrapolate", bounds_error=False)
                    yp[i][j] = f(xp)
        return xp, np.sum(np.sum(yp, axis=0), axis=0)




    def delta_q_calc(self, wavelength, r):
        '''Calculate the resolution function array,
        input: wavelength array and radius array,
        output: a two dimentional array with the corresponding sigma Q of the gaussian resolution in the cells'''
        LP = 1 / (1 / self.L1 + 1 / self.L2)
        delta_q = 1/12 * (2 * np.pi / wavelength) * (3 * (self.A1/2)**2 / self.L1**2 + 
                                                   3 * (self.A2/2)**2 / LP**2 +
                                                   (self.TubeWidth**2 + self.PixelHeight**2) / self.L2**2 + 
                                                   r[:, None]**2 / self.L2**2 * (self.DeltaLambdaRatio)**2)
        return delta_q
    
    def zero_divide(self, a, b):
        c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return c

    def moving_average(self, interval, WindowSize=10):
        window = np.ones(int(WindowSize)) / float(WindowSize)
        output = np.convolve(interval, window, 'same')
        return output

    def linear_func(self, k, b, x):
        return k * x + b
 
    # def trans_calc(self, data1, data2):
    #     StartFit = 20
    #     StopFit = int(len(data1) - 20)
    #     Data1Smoothed = self.moving_average(data1, WindowSize=3)
    #     Data2Smoothed = self.moving_average(data2, WindowSize=3)
    #     Div = self.zero_divide(Data1Smoothed, Data2Smoothed)
    #     X = np.linspace(StartFit, StopFit, StopFit-StartFit)
    #     Y = Div[StartFit:StopFit]
    #     c, stats = P.polyfit(X, Y, 3, full=True)
    #     # paras,err=scipy.optimize.curve_fit(lambda k,b,X: self.linear_func(k,b,X),X,Y,method = 'dogbox')
    #     DataX = np.arange(len(data1))
    #     DivFitedY = c[0] + c[1]*DataX + c[2]*DataX**2 + c[3]*DataX**3
    #     return DivFitedY

    def trans_calc(self, data1, data2):
        StartFit = 20
        StopFit = int(len(data1) - 20)
        Data1Smoothed = self.moving_average(data1, WindowSize=3)
        Data2Smoothed = self.moving_average(data2, WindowSize=3)
        Div = self.zero_divide(Data1Smoothed, Data2Smoothed)
        X = (data_reduce0.WavelengthArray)[StartFit:StopFit]  #np.linspace(StartFit,StopFit,StopFit-StartFit)
        Y = Div[StartFit:StopFit]
        c, stats = P.polyfit(X,Y,3,full=True)
        #paras,err=scipy.optimize.curve_fit(lambda k,b,X: self.linear_func(k,b,X),X,Y,method = 'dogbox')
        DataX = (self.WavelengthArray13[50] + self.WavelengthArray24[50])/2   #np.arange(len(data1))
        DivFitedY = c[0] + c[1]*DataX + c[2]*DataX**2 + c[3]*DataX**3
        return DivFitedY

   
#   def trans_calc(self,data1,data2):
#       StartFit = 20
#       StopFit = int(len(data1) - 20)
#       Data1Smoothed = self.moving_average(data1,WindowSize = 3)
#       Data2Smoothed = self.moving_average(data2,WindowSize = 3)
#       Div = self.zero_divide(Data1Smoothed,Data2Smoothed)
#       X = np.linspace(StartFit,StopFit,StopFit-StartFit)
#       Y = Div[StartFit:StopFit]
#       paras,err=scipy.optimize.curve_fit(lambda k,b,X: self.linear_func(k,b,X),X,Y,method = 'dogbox')
#       DataX = np.arange(len(data1))
#       DivFitedY = paras[1]*DataX + paras[0]
#       return DivFitedY  

    def plot_data(self,data):
        for t in range(len(data)):
            plt.plot(self.Q[:-5],data[t][:-5],label = 'D1' + str(t+1))
            plt.legend()
        plt.xlabel('Q (Å$^{-1}$)')
        plt.ylabel('Counts (n/Å)')
        plt.savefig('d4_Bank_plot2.svg',dpi = 600, format = 'svgz',bbox_inches = 'tight', transparent = True)
    
    def iq_plot(self,SampleTransNormed):
        QY = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        for t in range(len(SampleTransNormed)):
            if t ==0 or t == 2:
                for i in np.arange(0,self.RBins):  
                    for j in np.arange(0,self.WaveBins):
                        tmp = self.QBins - len(self.Q[self.Q > self.QArray13[i,j]]) - 1
                        QY[t][tmp] += SampleTransNormed[t,i,j]
            elif t == 1 or t ==3:
                for i in np.arange(0,self.RBins):  
                    for j in np.arange(0,self.WaveBins):
                        tmp = self.QBins - len(self.Q[self.Q > self.QArray24[i,j]]) - 1
                        QY[t][tmp] += SampleTransNormed[t,i,j]  
        self.plot_data(QY)
        return QY

    def linear_func(self,k,b,x):
        return k*x+b
    
    def data_fit(self,xx,yy):
        X = xx[yy>0]
        Y = yy[yy>0]
        paras,err=scipy.optimize.curve_fit(lambda k,b,X: self.linear_func(k,b,X),X,Y,method = 'dogbox')
        FitedY = paras[1]*xx + paras[0]
        return xx,FitedY  

    #def d_lambda_theta_calc(self,data):

    def add_error(self,x,y):
        return np.sqrt(x+y)

    def multiply_error(self,x,y):
        return np.sqrt((np.sqrt(x)*y)**2 + (np.sqrt(y)*x)**2)

    def divide_error(self,x,y):
        tmp1 = np.sqrt(x)/y
        tmp2 = np.sqrt(y)*x/y**2
        return np.sqrt(tmp1**2 + tmp2**2)

    #@jit(nopython=True)
    def translate_to_q(self,I0Smoothed13,I0Smoothed24,AirDirectPC,I0Scale,SampleTransNormed,SamplePC,SampleTrans,CellTransNormed,CellPC,CellTrans):
        global test
        WaveMin = self.StartWave
        bins24 = 1
        bins13 = 1
        WaveMax = self.StopWave
        SolidAngle = self.solid_angle()        
        I0Array13 = I0Smoothed13 
        I0Array24 = I0Smoothed24
        Normalization = {}
        QX = self.Q
        if QX[4]-QX[3] == QX[3]-QX[2]:
            QHalfBin = (QX[1:] + QX[:-1])/2
        else:
            QHalfBin = np.sqrt(QX[1:]*QX[:-1])
        Normalization[0] = SolidAngle[0][:,None]*I0Array13*self.SampleThickness/10*self.A2**2/self.A2Small**2
        Normalization[1] = SolidAngle[1][:,None]*I0Array24*self.SampleThickness/10*self.A2**2/self.A2Small**2
        Normalization[2] = SolidAngle[2][:,None]*I0Array13*self.SampleThickness/10*self.A2**2/self.A2Small**2
        Normalization[3] = SolidAngle[3][:,None]*I0Array24*self.SampleThickness/10*self.A2**2/self.A2Small**2
        DeltaQArray13 = self.delta_q_calc(self.WavelengthArray13[50],self.RArray)
        DeltaQArray24 = self.delta_q_calc(self.WavelengthArray24[50],self.RArray)
        QY = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        QYCell = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        QYNorm = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        DeltaQ = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        QSquare = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        QAve = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        CountSum = np.zeros(len(SampleTransNormed))[:,None]*np.zeros(self.QBins)
        for t in range(len(SampleTransNormed)):
            if t == 0 or t == 2:
                for i in np.arange(0,self.RBins - self.RBinsCut):  
                    for j in self.WaveBinsSelected: #np.arange(WaveMin,WaveMax):#(0,self.WaveBins):
                        CountIJ = SampleTransNormed[t,i,j]-CellTransNormed[t,i,j]
                        tmp = self.QBins - len(QHalfBin[QHalfBin > self.QArray13[i,j]]) - 1                        
                        QY[t][tmp] += SampleTransNormed[t,i,j]
                        QYCell[t][tmp] += CellTransNormed[t,i,j]
                        QYNorm[t][tmp] += Normalization[t][i,j]/bins13
                        DeltaQ[t][tmp] += DeltaQArray13[i,j]*CountIJ
                        QSquare[t][tmp] += self.QArray13[i,j]**2*CountIJ
                        QAve[t][tmp] += self.QArray13[i,j]*CountIJ
                        CountSum[t][tmp] += CountIJ
            elif t == 1 or t == 3:
                for i in np.arange(0,self.RBins - self.RBinsCut):  
                    for j in self.WaveBinsSelected: # np.arange(WaveMin,WaveMax):#(0,self.WaveBins):
                        CountIJ = SampleTransNormed[t,i,j]-CellTransNormed[t,i,j]
                        tmp = self.QBins - len(QHalfBin[QHalfBin > self.QArray24[i,j]]) - 1
                        QY[t][tmp] += SampleTransNormed[t,i,j] 
                        QYCell[t][tmp] += CellTransNormed[t,i,j]
                        QYNorm[t][tmp] += Normalization[t][i,j]/bins24
                        DeltaQ[t][tmp] += DeltaQArray24[i,j]*CountIJ
                        QSquare[t][tmp] += self.QArray24[i,j]**2*CountIJ
                        QAve[t][tmp] += self.QArray24[i,j]*CountIJ
                        CountSum[t][tmp] += CountIJ
                        
        QY = np.sum(QY,axis = 0)
        QYNorm = np.sum(QYNorm,axis = 0)
        QYCell = np.sum(QYCell,axis = 0)
        DeltaQ = np.sum(DeltaQ,axis = 0)
        CountSum = np.sum(CountSum,axis = 0)
        QSquare = np.sum(QSquare,axis = 0)
        QAve = np.sum(QAve,axis = 0)
        #QYError = 1/200*self.zero_divide(np.ones_like(QY),np.sqrt((QY*np.average(SampleTrans))**2 + (QYCell*np.average(CellTrans))**2)) + self.zero_divide(np.ones_like(QYNorm),QYNorm/I0Scale)
        #QYError = np.sqrt(QYError)
        #QYSampleCellError = 1/200*self.zero_divide(np.ones_like(QY),QY*np.average(SampleTrans)) + self.zero_divide(np.ones_like(QYNorm),QYNorm/I0Scale)
        #QYSampleCellError = np.sqrt(QYSampleCellError)
        #QYCellError = 1/200*self.zero_divide(np.ones_like(QY),QYCell*np.average(CellTrans)) + self.zero_divide(np.ones_like(QYNorm),QYNorm/I0Scale)
        #QYCellError = np.sqrt(QYCellError)

#        QYError = self.zero_divide(np.ones_like(QY),np.sqrt((QY*np.average(SampleTrans))**2 + (QYCell*np.average(CellTrans))**2)) + self.zero_divide(np.ones_like(QYNorm),QYNorm/I0Scale)
#        QYSampleCellError = self.zero_divide(np.ones_like(QY),QY*np.average(SampleTrans)) + self.zero_divide(np.ones_like(QYNorm),QYNorm/I0Scale)
#        QYCellError = self.zero_divide(np.ones_like(QY),QYCell*np.average(CellTrans)) + self.zero_divide(np.ones_like(QYNorm),QYNorm/I0Scale)
        #QYNorm = QYNorm
        GroupingScale = 2
        #tmp = self.zero_divide(np.ones_like(QY),QYCell*np.average(CellTrans))
        QYError = self.zero_divide(np.sqrt((np.sqrt(QY)/SamplePC*self.SampleScatteringFactor*GroupingScale)**2+(np.sqrt(QYCell)/CellPC*self.CellScatteringFactor*GroupingScale)**2),QYNorm/AirDirectPC) 

        QYSampleCellError = self.zero_divide((np.sqrt(QY)/SamplePC*self.SampleScatteringFactor*GroupingScale),QYNorm/AirDirectPC) 
        QYCellError = self.zero_divide((np.sqrt(QYCell)/CellPC*self.CellScatteringFactor*GroupingScale),QYNorm/AirDirectPC)


        #QYNorm = QYNorm/200
#        QYError = self.divide_error((self.add_error(QY*np.average(SampleTrans),QYCell*np.average(CellTrans)))**2,QYNorm*AirDirectPC/I0Scale)
#        QYSampleCellError = self.divide_error(QY*np.average(SampleTrans),QYNorm*AirDirectPC/I0Scale)
#        QYCellError = self.divide_error(QYCell*np.average(CellTrans),QYNorm*AirDirectPC/I0Scale)

        #QYNorm = QYNorm/200
        QYSample = self.zero_divide((QY/SamplePC*self.SampleScatteringFactor-QYCell/CellPC*self.CellScatteringFactor),QYNorm/AirDirectPC)
        QYSampleCell = self.zero_divide((QY/SamplePC*self.SampleScatteringFactor),QYNorm/AirDirectPC)
        QYCell = self.zero_divide((QYCell/CellPC*self.CellScatteringFactor),QYNorm/AirDirectPC)
        
        QXError = np.sqrt(np.abs(self.zero_divide(DeltaQ,CountSum) + self.zero_divide(QSquare,CountSum) - (self.zero_divide(QAve,CountSum))**2))
        #QXError = (self.zero_divide(DeltaQ,CountSum) + self.zero_divide(QSquare,CountSum) - (self.zero_divide(QAve,CountSum))**2)/1
        QXError = self.denan(QXError)
        return self.Q,QYSample,QYError,QXError, QYSampleCell, QYSampleCellError, QYCell, QYCellError

  
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


    def save_file(self,xx,yy,zz,file_name):
        with open(file_name,'w') as f:
            
            print('{:<20.12s}{:>20.12s}{:>20.12s}'.format('Q','I(Q)','Sigma I(Q)'),file = f)
            for x,y,z in zip(xx,yy,zz):
                print('{:<20.8f}{:>20.8f}{:>20.8f}'.format(x,y,z),file = f)
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

    def save_file40(self,xx,yy,zz,tt,file_name,ExperimentTime):
        with open(file_name,'w') as f:
            print('Problems or bugs please contact Taisen Zuo (zuots@ihep.ac.cn) ' + 'Experiment started at ' + str(ExperimentTime[0]) + ' and run ' + str(round(ExperimentTime[3],2)) + ' minutes. ' + 'File created:' + str(self.formated_time),file = f)
            print('{:<20.12s}{:<20.12s}{:<20.12s}{:<20.12s}'.format('Q','I(Q)','Sigma I(Q)','Sigma Q'),file = f)
            for x,y,z,t in zip(xx,yy,zz,tt):
                print('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(x,y,z,t),file = f)
        f.close()


    def save_file4(self,xx,yy,zz,tt,file_name,ExperimentTime):
        tt = [x if isinstance(x, (int, float)) else 0 for x in tt]
        with open(file_name,'w') as f:
            print('Problems or bugs please contact Taisen Zuo (zuots@ihep.ac.cn) ' + 'Experiment started at ' + str(ExperimentTime[0]) + ' and run ' + str(round(ExperimentTime[3],2)) + ' minutes. ' + 'File created:' + str(self.formated_time),file = f)
            print('{:<20.12s}{:<20.12s}{:<20.12s}{:<20.12s}'.format('Q','I(Q)','Sigma I(Q)','Sigma Q'),file = f)
            for x,y,z,t in zip(xx,yy,zz,tt):
                print('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(x,y,z,t),file = f)
        f.close()

    def save_file4_igor(self,xx,yy,zz,tt,file_name,ExperimentTime):
        tt = [x if isinstance(x, (int, float)) else 0 for x in tt]
        with open(file_name,'w') as f:
            print('Experiment started at ' + str(ExperimentTime[0]) + ' and run ' + str(ExperimentTime[3]) + ' minutes. ' + 'COMBINED FILE CREATED:' + str(self.formated_time),file = f)
            print('''NSORT-ed   MAR16246.ABS 	  +  MAR16236.ABS	  + MAR16226.ABS 
normalized to   MAR16236.ABS 
multiplicative factor 1-2 =            1	 multiplicative factor 2-3 = 1 
The 6 columns are | Q (1/A) | I(Q) (1/cm) | std. dev. I(Q) (1/cm) | sigmaQ | meanQ | ShadowFactor|''',file = f)
           # print('{:<20.12s}{:<20.12s}{:<20.12s}{:<20.12s}'.format('Q','I(Q)','Sigma I(Q)','Sigma Q'),file = f)
            tt_one = np.ones_like(xx)
            for x,y,z,t,t_one in zip(xx,yy,zz,tt,tt_one):
                print('{:>15.5f}{:>16.2f}{:>16.2f}{:>16.2f}{:>16.2f}{:>16.2f}'.format(x,y,z,t,x,t_one),file = f)
        f.close()

    def plot_data(self,xx,yy,label = np.arange(1,100),save = False, FileName = 'SavedPlot.svg', title = 'title', xlabel = 'Lambda (Å)', ylabel = 'Counts (n/s/Å)',show = False, logx = False, logy = False): 
        lines = len(xx)
        plt.yscale('symlog', linthresh=0.001)
        for i in range(lines):#(len(tof)):
        #    if lines <3:
        #        yy[i][yy[i]<=0] = np.min(yy[i][yy[i]>0])
        #    elif lines >=3:
        #    yy[i][yy[i]<=0] = 1E-4
            plt.plot(xx[i],yy[i],label=label[i])
        plt.rcParams.update({'font.size': 15})
        #plt.rc('font',family = 'Times New Roman')
        #plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        plt.tick_params(axis = 'both',direction = 'in',labelsize = 15,width = 2)
        plt.tick_params(axis = 'both',which = 'minor',direction = 'in',labelsize = 15,width = 1.5,length = 4)
        plt.legend(fancybox=True, framealpha=0.01,fontsize = 16,frameon = False,loc = 'upper right')
        plt.xlabel(xlabel,size = 20)
        plt.ylabel(ylabel,size =20)
        #plt.ylim = (-50,50)
        #plt.grid(True, which="both", ls="--")
        #plt.tight_layout()
        plt.title(title)
        if logx is True:
            plt.xscale('log')
        if logy is True:
            plt.yscale('log')
        if save is True:
            plt.savefig(FileName,dpi = 600, format = 'svgz',bbox_inches = 'tight', transparent = True)
        if show == True:
            plt.show()
        plt.close()

    def plot_data_with_errorbar(self,xx,yy,error,label = np.arange(1,100),save = False, FileName = 'SavedPlot.svg', title = 'title', xlabel = 'Lambda (Å)', ylabel = 'Counts (n/s/Å)', logx = False, logy = False,show = False):
        common_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'white', 'orange', 'purple']
        #print(len(xx),len(yy))
        lines = len(xx)
        plt.figure(figsize=(20, 13))
        plt.yscale('symlog', linthresh=1E-4)
        yy2 = yy.copy()
        for i in range(lines):#(len(tof)):
            #print(len(xx),len(yy))
            #if lines <3:
            #    yy[i][yy[i]<=0] = np.min(yy[i][yy[i]>0])
            #elif lines >=3:
            #yy[i][yy[i]<=0] = 1E-4
            #plt.plot(xx[i],yy[i],label=label[i])
            #yy2[i][yy2[i]<=0] = 5E-4
            plt.errorbar(xx[i],yy2[i],np.abs(error[i]),label=label[i],fmt='-', color= common_colors[i],ecolor = common_colors[i],  elinewidth=1, capsize=3)
        plt.rcParams.update({'font.size': 30})
        #plt.rc('font',family = 'Times New Roman')
        #plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        plt.tick_params(axis = 'both',direction = 'in',labelsize = 30,width = 2)
        plt.tick_params(axis = 'both',which = 'minor',direction = 'in',labelsize = 30, width = 2.5,length = 4)
        plt.legend(fancybox=True, framealpha=0.01,fontsize = 16,frameon = False,loc = 'upper right')
        plt.xlabel(xlabel,size = 40)
        plt.ylabel(ylabel,size = 40)
        #plt.ylim = (-50,50)
        plt.title(title)
        #plt.grid(True, which="both", ls="--")
        if logx is True:
            plt.xscale('log')
        if logy is True:
            plt.yscale('log')
        if save is True:
            plt.savefig(FileName,dpi = 600, format = 'svgz',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
        plt.close()


    def trans_plot(self,yy, labels = ['Trans1','Trans2']):
        xx = (self.WavelengthArray13[50] + self.WavelengthArray24[50])/2
        myplot222.plot(xx,yy,xlabel = 'Neutron wavelength (A)', ylabel = 'Transmission', labels = labels, save = True, save_name = str("Data") +'_trans.svg')
            
    def save_image(self,xx,yy,ImageName):
        myplot_log_log.plot(xx,yy,xlabel = 'Q (Å$^{-1}$)', ylabel = 'I (cm$^{-1}$)', save = True, save_name = ImageName)

    def save_image_with_error_bar(self,xx,yy,ErrorBar,ImageName,label):
        myplot222.plot(xx,yy,ErrorBar,xlabel = 'Q (Å$^{-1}$)', ylabel = 'I (cm$^{-1}$)', label = label,save = True, save_name = ImageName)

    def I0_interp(self, data,func):
        out = np.zeros_like(data)
        for i in range(len(data)):
            out[i] = func(data[i])
        return out
    
    def read_exp_data(self,FileName):
        with open(FileName,"r") as f:
            Q = []
            IQ = []
            SigmaIQ = []
            SigmaQ = []
            for line in f.readlines()[2:]:
                sline = line.split()
                Q.append(float(sline[0]))
                IQ.append(float(sline[1]))
                SigmaIQ.append(float(sline[2]))
                SigmaQ.append(float(sline[3]))
        Q = np.array(Q)
        IQ = np.array(IQ)
        SigmaIQ = np.array(SigmaIQ)
        SigmaQ = np.array(SigmaQ)
        return Q,IQ,SigmaIQ,SigmaQ


    def de_none(self,arr):
        for i in range(len(arr)):
            if arr[i] is None or (isinstance(arr[i], float) and np.isnan(arr[i])):
                if i == 0:  # 如果第一个值是None或NaN，设置为0
                    arr[i] = 0
                else:
                    arr[i] = arr[i - 1]  # 用前一个有效值替代
        return arr

    def gauss_smear_array(self,qx,iqx,qxp,px):
        X,Y = self.gauss_array(qxp,px)
        func = np.interp(qx[:,None]-X,qx,iqx,left=iqx[0],right = iqx[-1])*Y
        correlation = np.trapz(func,X)#*len(qx)
        return correlation

    def gauss_array(self,gq,sigmaQ):
        n = len(sigmaQ)  # 曲线数量 (120)
        m = len(gq)         # 每条曲线的点数
        
        # 初始化输出数组
        X = np.zeros((n, m))
        Y = np.zeros((n, m))
        
        # 为每个sigmaQ生成高斯曲线
        for i, sigma in enumerate(sigmaQ):
            # 生成x值范围：-3σ 到 3σ
            x = np.linspace(-4*sigma, 4*sigma, m)
            # 计算归一化高斯函数 (概率密度函数)
            if sigma <= 0 or sigma is None or np.isnan(sigma):
                y = np.zeros(m)
            else:
                y = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
            
            # 存储结果
            X[i] = x
            Y[i] = y
        #Y = self.de_none_2D(Y)
        #print(Y)
        return X,Y

    def vcg_gauss_desmear_array(self,x,y,qxp,gsigma):
        beta = 1
        y1 = y - beta*(self.gauss_smear_array(x,y,qxp,gsigma) - y)
        tmp = []
        index = []
        index.append(0)
        tmp.append(y1)
        for i in range(100):
            yn = tmp[i] - beta*(self.gauss_smear_array(x,tmp[i],qxp,gsigma) - y)
            tmp.append(yn)
            #tmp[i] = 0
            
            index0 = (abs(y/self.gauss_smear_array(x,tmp[i],qxp,gsigma) - 1))
            index0 = self.de_none(index0)
            index.append(np.average(index0))
            if np.average(index[i+1]) > np.average(index[i]):
                beta = beta/1.2
            #print(np.average(index))
            if np.average(index0) < 0.05:
                print('desmear interation times i=',i)
                return yn             
        print(np.average(index[i]))   
        print('desmear interation times i=',i+1)
        return yn  

    def vcg_gauss_desmear(self,x,y,gsigma):
        beta = 1
        y1 = y - beta*(self.gauss_smear_y(x,y,gsigma) - y)
        tmp = []
        index = []
        index.append(0)
        tmp.append(y1)
        for i in range(20):
            yn = tmp[i] - beta*(self.gauss_smear_y(x,tmp[i],gsigma) - y)
            tmp.append(yn)
            #tmp[i] = 0
            
            index0 = (abs(y/self.gauss_smear_y(x,tmp[i],gsigma) - 1))
            index.append(np.average(index0))
            if np.average(index[i+1]) > np.average(index[i]):
                beta = beta/1.2
            #print(np.average(index))
            if np.average(index0) < 0.05:
                print('i=',i)
                return yn  
        print('desmear interation times i=',i+1)              
        return yn   

def zero_divide(a,b):
    c = np.divide(a,b,out = np.zeros_like(a), where = b!=0)
    return c

def de_none(arr):
    for i in range(len(arr)):
        if arr[i] is None or (isinstance(arr[i], float) and np.isnan(arr[i])):
            if i == 0:  # 如果第一个值是None或NaN，设置为0
                arr[i] = 0
            else:
                arr[i] = arr[i - 1]  # 用前一个有效值替代
    return arr

def reduce(samples,info):
    global data_reduce
    global data_reduce0
    global D3
    DataFold = info.DataFold
    #RunInfoFold = info.RunInfoFold
    data_reduce = data_reduce(DataFold,info)
    data_reduce0 = data_reduce_D3.data_reduce(DataFold,info)
    #tmp = data_reduce.TimeDelay
    #data_reduce0.TimeDelay = tmp + 2 
    SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness = samples
    reduce_one_data(SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness,info)

def get_run_fold(run_num):
    run_fold = r"RUN" + str('0'*(7-len(run_num.split('_')[0]))) + run_num
    return run_fold

def reduce_one_data(SampleName,SampleScattering,CellScattering,SampleDirect,CellDirect,AirDirect,SampleThickness,info):
    global data_reduce0
    SampleScatteringRun = get_run_fold(SampleScattering) #r"RUN" + str('0'*(7-len(SampleScattering))) + SampleScattering
    SampleDirectRun =  get_run_fold(SampleDirect) #r"RUN"  + str('0'*(7-len(SampleDirect))) + SampleDirect
    CellScatteringRun =  get_run_fold(CellScattering) #r"RUN"  + str('0'*(7-len(CellScattering))) + CellScattering
    CellDirectRun =  get_run_fold(CellDirect) #r"RUN"  + str('0'*(7-len(CellDirect))) + CellDirect
    AirDirectRun =  get_run_fold(AirDirect) #r"RUN"  + str('0'*(7-len(AirDirect))) + AirDirect


    data_reduce.SampleThickness = float(SampleThickness)
    data_reduce0.SampleThicknes = float(SampleThickness)
    AirDirect,AirDirectPC = data_reduce0.load_data(AirDirectRun)
    CellDirect,CellDirectPC = data_reduce0.load_data(CellDirectRun)
    SampleDirect,SampleDirectPC = data_reduce0.load_data(SampleDirectRun)

    CellScattering,CellPC = data_reduce.load_data(CellScatteringRun) #*0
    SampleScattering,SamplePC = data_reduce.load_data(SampleScatteringRun)

    ExperimentTimeSample = data_reduce.get_experimental_time_info(SampleScatteringRun)
    ExperimentTimeCell = data_reduce.get_experimental_time_info(CellScatteringRun)

    AirDirect = AirDirect#/200 #/AirDirectPC
    CellDirect = CellDirect#/200 #*info.DirectFactorCell/CellDirectPC
    SampleDirect = SampleDirect#/200 #*info.DirectFactorSample/SampleDirectPC


    I0X,I0Y = data_reduce0.direct_beam_integrate_to_lambda(AirDirect)
    ISX,ISY = data_reduce0.direct_beam_integrate_to_lambda(SampleDirect)
    ICX,ICY = data_reduce0.direct_beam_integrate_to_lambda(CellDirect)

    SampleTrans = data_reduce.trans_calc(ISY*info.DirectFactorSample/SampleDirectPC,I0Y/AirDirectPC)
    CellTrans = data_reduce.trans_calc(ICY*info.DirectFactorCell/CellDirectPC,I0Y/AirDirectPC)

###################################

#   ISSD1X,ISSD1Y = data_reduce.direct_beam_integrate_to_lambda(data_reduce.WavelengthArray13[0],SampleScattering)
#   ICSD1X,ICSD1Y = data_reduce.direct_beam_integrate_to_lambda(data_reduce.WavelengthArray24[0],CellScattering)
#
#   data_reduce.save_file2(ISSD1X,ISSD1Y,info.OutPath + '/' + "I_Sample_Scattering_Lambda_D1" + SampleName + ".dat")
#   data_reduce.save_file2(ICSD1X,ICSD1Y,info.OutPath + '/' + "I_Cell_Scattering_Lambda_D1" + SampleName + ".dat")
#   data_reduce.data_plot_xy(ISSD1X,ISSD1Y,label='I_Sample_Scattering_Lambda plot_D1', show = False,FileName = info.OutPath + '/'+ "I_Sample_Scattering_Lambda_D1" + SampleName)
#   data_reduce.data_plot_xy(ICSD1X,ICSD1Y,label='I_Cell_Scattering_Lambda plot_D1', show = False ,FileName = info.OutPath + '/'+ "I_Cell_Scattering_Lambda_D1" + SampleName)

###################################

    Sec2Theta13 = 1/np.cos((data_reduce.ThetaArray13[:,None]))
    Sec2Theta24 = 1/np.cos((data_reduce.ThetaArray24[:,None]))
    Sec2Theta = (Sec2Theta13 + Sec2Theta24)/2
    SampleScattering[0] = SampleScattering[0]/(SampleTrans**((1+Sec2Theta13)/2))
    SampleScattering[2] = SampleScattering[2]/(SampleTrans**((1+Sec2Theta13)/2))
    SampleScattering[1] = SampleScattering[1]/(SampleTrans**((1+Sec2Theta24)/2))
    SampleScattering[3] = SampleScattering[3]/(SampleTrans**((1+Sec2Theta24)/2))

    CellScattering[0] = CellScattering[0]/(CellTrans**((1+Sec2Theta13)/2))
    CellScattering[2] = CellScattering[2]/(CellTrans**((1+Sec2Theta13)/2))
    CellScattering[1] = CellScattering[1]/(CellTrans**((1+Sec2Theta24)/2))
    CellScattering[3] = CellScattering[3]/(CellTrans**((1+Sec2Theta24)/2))

    SampleTransNormed = SampleScattering/(SampleTrans**((1+Sec2Theta)/2))
    CellTransNormed =  CellScattering/(CellTrans**((1+Sec2Theta)/2)) # 

    I0 = I0Y*data_reduce.I0Scale
    I0XStack = np.hstack((I0X - data_reduce.TOF, I0X, I0X + data_reduce.TOF))
    I0YStack = np.hstack((np.zeros_like(I0Y),I0,np.zeros_like(I0Y)))
    I0InterpolatedToD1_13 = np.interp(data_reduce.WavelengthArray13,I0XStack, I0YStack,left = 0,right =0)
    I0InterpolatedToD1_24 = np.interp(data_reduce.WavelengthArray24,I0XStack, I0YStack,left = 0,right = 0)



    if info.debug2 is True and info.debug is True:
    
        #print('I_DirectBeamSample_D3:',np.sum(data_reduce0.load_data(SampleDirectRun)))
        #print('I_DirectBeamSample_D1:',np.sum(data_reduce.load_data(SampleDirectRun)))
        #print('I_SampleScattering_D3:',np.sum(data_reduce0.load_data(SampleScatteringRun)))
        #print('I_SampleScattering_D1:',np.sum(data_reduce.load_data(SampleScatteringRun)))
       # print('wavelength:',data_reduce.WavelengthArray13[10].shape)
       # print('I0Interpolated:',I0InterpolatedToD1_13.shape)
       # print('sample:',np.sum(SampleTransNormed,axis = 0).shape)
        I0Sumed = np.sum(0.1*I0InterpolatedToD1_13[50]/data_reduce.I0Scale)
        Sample0Sumed = np.sum(np.sum(SampleTransNormed,axis = 0)[0:10]) 
        Sample50Sumed = np.sum(np.sum(SampleTransNormed,axis = 0)[45:55])
        Sample99Sumed = np.sum(np.sum(SampleTransNormed,axis = 0)[89:99])
        plt.show()    
        plt.close()
        plt.plot(data_reduce.WavelengthArray13[50],0.1*I0InterpolatedToD1_13[50]/data_reduce.I0Scale,label = 'I(Lambda)AirDirect')
        plt.plot(data_reduce.WavelengthArray13[50],I0Sumed/Sample50Sumed*np.sum(np.sum(SampleTransNormed,axis = 0)[45:55],axis = 0),label='I(Lambda)SampleScatteringD1_Rmiddle')
        plt.plot(data_reduce.WavelengthArray13[0],I0Sumed/Sample0Sumed*np.sum(np.sum(SampleTransNormed,axis = 0)[0:10],axis = 0),label='I(Lambda)SampleScatteringD1_Rmin')
        plt.plot(data_reduce.WavelengthArray13[99],I0Sumed/Sample99Sumed*np.sum(np.sum(SampleTransNormed,axis = 0)[89:99],axis = 0),label='I(Lambda)SampleScatteringD1_Rmax')
        plt.legend(fancybox=True, framealpha=0.01,frameon = False)
        plt.title(SampleName)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Counts (n/s/Å)')
        plt.xscale('linear')
        plt.yscale('log')
       # plt.xlim(1,15)
        plt.show()
        plt.close()

    QXP,QYP,QYError,QXError,QYSample,QYSampleError,QYCell,QYCellError = data_reduce.translate_to_q(I0InterpolatedToD1_13,I0InterpolatedToD1_24,AirDirectPC,data_reduce.I0Scale,SampleTransNormed,SamplePC,SampleTrans,CellTransNormed,CellPC,CellTrans)
#    data_reduce.save_file(QXP,QYP,QYError,info.OutPath + '/' + "IQ_NormedD2_" + SampleName + ".dat")

    if info.ExtraBkg > 0:
        QYCell = QYCell - info.ExtraBkg

    data_reduce.save_file40(QXP,QYP,QYError,QXError,info.OutPath + '/' + "IQ_Normed_with_QXErrorD1_OnlySample_" + SampleName + ".dat",ExperimentTimeSample)
    data_reduce.save_file40(QXP,QYSample,QYSampleError,QXError,info.OutPath + '/' + "IQ_Normed_with_QXErrorD1_Sample+Cell_" + SampleName + ".dat",ExperimentTimeSample)
    data_reduce.save_file40(QXP,QYCell,QYCellError,QXError,info.OutPath + '/' + "IQ_Normed_with_QXErrorD1_OnlyCell_" + SampleName + ".dat",ExperimentTimeCell)

#    QXP,QYP,QYError,QXError = data_reduce.translate_to_q(I0InterpolatedToD1_13,I0InterpolatedToD1_24,AirDirectPC,data_reduce.I0Scale,SampleTransNormed,SamplePC,SampleTrans,CellTransNormed,CellPC,CellTrans)

#    data_reduce.save_file(QXP,QYP,QYError,info.OutPath +'/' + 'IQ_NormedD1_' + SampleName + ".dat")
    #data_reduce.save_image_with_error_bar(QXP,QYP,QYError/2,info.OutPath +'/' + "IQ_NormedD1" + SampleName + ".svg", label = SampleName)
#    data_reduce.save_file4(QXP,QYP,QYError,QXError,info.OutPath + '/' + 'IQ_Normed_with_QXErrorD1_' + SampleName + ".dat",ExperimentTimeSample)
    #
    #data_reduce0.save_file2(data_reduce0.WavelengthArray,SampleTrans,info.OutPath + 'Sample_TransD1' + SampleName + ".dat")
    #data_reduce0.save_file2(data_reduce0.WavelengthArray,CellTrans,info.OutPath + 'Cell_TransD1" + SampleName + ".dat")
    #
    #data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(AirDirect,axis = 0),"AirDirect_Lambda" + SampleName + ".dat")
    #data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(SampleDirect,axis = 0),"SampleDirect_Lambda" + SampleName + ".dat")
    #data_reduce.save_file2(data_reduce.WavelengthArray,np.sum(SampleScattering,axis = 0),"SampleScattering_Lambda" + SampleName + ".dat")
    
    
    
    FileName = SampleName
    
    Q1,IQ1,SigmaIQ1,SigmaQ1 = data_reduce.read_exp_data(info.OutPath + '/' + 'IQ_Normed_with_QXErrorD1_OnlySample_'+  FileName + '.dat')
    Q2,IQ2,SigmaIQ2,SigmaQ2 = data_reduce.read_exp_data(info.OutPath + '/' +'IQ_Normed_with_QXErrorD2_OnlySample_'+  FileName + '.dat')
    Q3,IQ3,SigmaIQ3,SigmaQ3 = data_reduce.read_exp_data(info.OutPath + '/' + 'IQ_Normed_with_QXErrorD3_OnlySample_'+  FileName + '.dat')

    Q1,IQ1Sample,SigmaIQ1Sample,SigmaQ1 = data_reduce.read_exp_data(info.OutPath + '/' + 'IQ_Normed_with_QXErrorD1_Sample+Cell_'+  FileName + '.dat')
    Q2,IQ2Sample,SigmaIQ2Sample,SigmaQ2 = data_reduce.read_exp_data(info.OutPath + '/' +'IQ_Normed_with_QXErrorD2_Sample+Cell_'+  FileName + '.dat')
    Q3,IQ3Sample,SigmaIQ3Sample,SigmaQ3 = data_reduce.read_exp_data(info.OutPath + '/' + 'IQ_Normed_with_QXErrorD3_Sample+Cell_'+  FileName + '.dat')

    Q1,IQ1Cell,SigmaIQ1Cell,SigmaQ1 = data_reduce.read_exp_data(info.OutPath + '/' + 'IQ_Normed_with_QXErrorD1_OnlyCell_'+  FileName + '.dat')
    Q2,IQ2Cell,SigmaIQ2Cell,SigmaQ2 = data_reduce.read_exp_data(info.OutPath + '/' +'IQ_Normed_with_QXErrorD2_OnlyCell_'+  FileName + '.dat')
    Q3,IQ3Cell,SigmaIQ3Cell,SigmaQ3 = data_reduce.read_exp_data(info.OutPath + '/' + 'IQ_Normed_with_QXErrorD3_OnlyCell_'+  FileName + '.dat')


    if info.igor_format == True:
        data_reduce.save_file4_igor(Q1,IQ1,SigmaIQ1,SigmaQ1,info.OutPath +'/' + SampleName + "OnlySample_D1_igor.dat",ExperimentTimeSample)
        data_reduce.save_file4_igor(Q2,IQ2,SigmaIQ2,SigmaQ2,info.OutPath +'/' + SampleName + "OnlySample_D2_igor.dat",ExperimentTimeSample)   
        data_reduce.save_file4_igor(Q3,IQ3,SigmaIQ3,SigmaQ3,info.OutPath +'/' + SampleName + "OnlySample_D3_igor.dat",ExperimentTimeSample) 

    if info.debug is True:
        data_reduce.plot_data_with_errorbar([Q1,Q2,Q3,Q1,Q2,Q3],[IQ1Sample,IQ2Sample,IQ3Sample,IQ1Cell,IQ2Cell,IQ3Cell],[SigmaIQ1Sample,SigmaIQ2Sample,SigmaIQ3Sample,SigmaIQ1Cell,SigmaIQ2Cell,SigmaIQ3Cell],label = ['D1_Sample+Cell','D2_Sample+Cell','D3_Sample+Cell','D1_Cell','D2_Cell','D3_Cell'],title = FileName,xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True,logy = True, save = False, FileName = info.OutPath +'/' + FileName + 'SampleCell_combined.svg',show = True) 
        data_reduce.plot_data_with_errorbar([Q1,Q2,Q3],[IQ1,IQ2,IQ3],[SigmaIQ1,SigmaIQ2,SigmaIQ3],label = ['D1','D2','D3'],title = FileName,xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath +'/' + FileName + 'Sample-Cell_combined.svg',show = True)
    data_reduce.plot_data_with_errorbar([Q1,Q2,Q3,Q1,Q2,Q3],[IQ1Sample,IQ2Sample,IQ3Sample,IQ1Cell,IQ2Cell,IQ3Cell],[SigmaIQ1Sample,SigmaIQ2Sample,SigmaIQ3Sample,SigmaIQ1Cell,SigmaIQ2Cell,SigmaIQ3Cell],label = ['D1_Sample+Cell','D2_Sample+Cell','D3_Sample+Cell','D1_Cell','D2_Cell','D3_Cell'],title = FileName,xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True,logy = True, save = True,FileName = info.OutPath +'/' + FileName + 'SampleCell_combined.svg',show = False)    
    data_reduce.plot_data_with_errorbar([Q1,Q2,Q3],[IQ1,IQ2,IQ3],[SigmaIQ1,SigmaIQ2,SigmaIQ3],label = ['D1','D2','D3'],title = FileName,xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath +'/' + FileName + 'Sample-Cell_combined.svg',show = False)
    def linear_func(k,b,x):
        return k*x+b
 
    def find_stitch_point(a,b,GuessPoint):
        FitRange =  10
        SearchRange = 10
        FOM = 1E7
        StitchPoint = GuessPoint
        for i in range(GuessPoint-SearchRange,GuessPoint+SearchRange):
            if b[i]>0:
                bp = (b*a[i]/b[i]).copy()
                arr = np.hstack((a[(GuessPoint-FitRange):GuessPoint],bp[GuessPoint:(GuessPoint+FitRange)]))
                xx = np.arange(0,len(arr))
                c, stats = P.polyfit(xx,arr,1,full=True)
                tmp = stats[0][0]
                if tmp < FOM:
                    FOM = tmp
                    StitchPoint = i     
        return StitchPoint 
   
    def adjust_point(point,IQ1,IQ2):
        div = []
        for i in range(0,2):
            tmp = IQ1[point + i] - IQ2[point + i]
            div.append(abs(tmp))
        div = np.array(div)
        # 找到最小值
        min_value = min(div)
        # 获取最小值的下标
        min_index = 0
        for i in range(len(div)):
            if div[i] == min_value:
                min_value = div[i]
                min_index = i
        return point + min_index

    def extend_IQ(x2,y2,SigmaQ):
        for i in range(len(y2)):
            if y2[i]>0:
                t = i
                break
        k, b = np.polyfit(x2[t:t+5], y2[t:t+5], 1)
        y2[:t] = k*x2[:t]+b
        SigmaQ[:t] = SigmaQ[t]
        return x2,y2,SigmaQ,t

    def stitch_data(Q1,Q2,Q3,IQ1,IQ2,IQ3,SigmaIQ1,SigmaIQ2,SigmaIQ3,SigmaQ1,SigmaQ2,SigmaQ3):
        global data_reduce
        Q = Q1
        #exbkg = 0.02
        #IQ1,IQ2,IQ3 = IQ1-exbkg,IQ2-exbkg,IQ3-exbkg 
        IQ = np.zeros(len(IQ1))
        SigmaIQ = np.zeros(len(SigmaIQ1))
        SigmaQ = np.zeros(len(SigmaQ1))
#        Point1 = len(Q[Q < 2.2/info.WaveMin*0.12]) #2.2/   0.13  0.11 
#        Point2 = len(Q[Q < 2.2/info.WaveMin*0.41])-1  # 2.2/  0.4 0.41
#        Point1 = find_stitch_point(IQ3,IQ2,Point1)
#        Point2 = find_stitch_point(IQ2,IQ1,Point2)
        Point1 = len(Q[Q <= info.StitchPoint1])
        Point2 = len(Q[Q <= info.StitchPoint2])
        #print('beforeAdjust',Point1,Point2,info.StitchPoint1,info.StitchPoint2)
        Point1 = adjust_point(Point1,IQ3,IQ2)
        Point2 = adjust_point(Point2,IQ2,IQ1)
        #print('AfterAdjust:',Point1,Point2)
        #print('Fit point:', Point11,Point22)
        #Point2 = 79
       # Point2 = find_stitch_point(IQ1,IQ2)
        #Point1 = 62
        #Point1 = find_stitch_point(IQ2,IQ3)
        #print('Point1',Point1,'Point2',Point2)
        if info.adjustD2 == True:
            if np.sum(IQ2[Point1-1:Point1+1]) > 1E-5:
                IQ2 = IQ2*np.sum(IQ3[Point1-1:Point1+1])/np.sum(IQ2[Point1-1:Point1+1])
                SigmaIQ2 = SigmaIQ2*np.sum(IQ3[Point1-1:Point1+1])/np.sum(IQ2[Point1-1:Point1+1])
        if info.adjustD1 == True:
            if np.sum(IQ1[Point2-1:Point2+1]) > 1E-5:
                IQ1 = IQ1*np.sum(IQ2[Point2-1:Point2+1])/np.sum(IQ1[Point2-1:Point2+1])
                SigmaIQ1 = SigmaIQ1*np.sum(IQ2[Point2-1:Point2+1])/np.sum(IQ1[Point2-1:Point2+1])
        if data_reduce.IncludeD1 is False:
            IQ = np.hstack((IQ3[:Point1],IQ2[Point1:]))
            SigmaIQ = np.hstack((SigmaIQ3[:Point1],SigmaIQ2[Point1:]))
            SigmaQ = np.hstack((SigmaQ3[:Point1],SigmaQ2[Point1:]))
        else:
            IQ = np.hstack((IQ3[:Point1],IQ2[Point1:Point2],IQ1[Point2:]))
            SigmaIQ = np.hstack((SigmaIQ3[:Point1],SigmaIQ2[Point1:Point2],SigmaIQ1[Point2:]))
            SigmaQ = np.hstack((SigmaQ3[:Point1],SigmaQ2[Point1:Point2],SigmaQ1[Point2:]))
        return Q, IQ, SigmaIQ, SigmaQ
    #Q, IQ, SigmaIQ, SigmaQ = stitch_data(Q1,Q2,Q3,IQ1,IQ2,IQ3,SigmaIQ1,SigmaIQ2,SigmaIQ3,SigmaQ1,SigmaQ2,SigmaQ3)
    Q, IQSample, SigmaIQSample, SigmaQ = stitch_data(Q1,Q2,Q3,IQ1Sample,IQ2Sample,IQ3Sample,SigmaIQ1Sample,SigmaIQ2Sample,SigmaIQ3Sample,SigmaQ1,SigmaQ2,SigmaQ3)    
    Q, IQCell, SigmaIQCell, SigmaQ = stitch_data(Q1,Q2,Q3,IQ1Cell,IQ2Cell,IQ3Cell,SigmaIQ1Cell,SigmaIQ2Cell,SigmaIQ3Cell,SigmaQ1,SigmaQ2,SigmaQ3)

    if info.StitchAndThenSubCell is True:
        IQ = IQSample - IQCell
        SigmaIQ = np.sqrt(SigmaIQSample**2 + SigmaIQCell**2)
    else:
        Q, IQ, SigmaIQ, SigmaQ = stitch_data(Q1,Q2,Q3,IQ1,IQ2,IQ3,SigmaIQ1,SigmaIQ2,SigmaIQ3,SigmaQ1,SigmaQ2,SigmaQ3)

#######################################save data and  figures
    if info.debug is True: 
        data_reduce.plot_data_with_errorbar([Q],[IQ],[SigmaIQ],label = [FileName+'stitched'], title = FileName, xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath +'/'+ 'StitchedDataOnlySample' +'/' + FileName + 'stitched.svg',show = True)
   #     data_reduce.plot_data_with_errorbar([Q],[SigmaQ],[SigmaQ]/20,label = [FileName+'stitched'], title = FileName, xlabel = 'Q (Å$^{-1}$)',ylabel = 'SigmaQ (Å$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath +'/' + FileName + 'stitched.svg',show = True)

    if np.any(IQ) > 0 and info.DesmearData is False and info.LambdaDivide is False:
        data_reduce.plot_data_with_errorbar([Q],[IQ],[SigmaIQ],label = [FileName+'OnlySample_stitched_'], title = FileName, xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath +'/' + 'StitchedDataOnlySample' +'/' + FileName + 'OnlySample_stitched.svg')
    data_reduce.save_file4(Q,IQ,SigmaIQ,SigmaQ,info.OutPath +'/' + 'StitchedDataOnlySample' + '/' + SampleName + "OnlySample_Stitched.dat",ExperimentTimeSample)

    if info.igor_format == True:
        data_reduce.save_file4_igor(Q,IQ,SigmaIQ,SigmaQ,info.OutPath +'/' + 'StitchedDataOnlySample' + '/' + SampleName + "OnlySample_Stitched_igor.dat",ExperimentTimeSample)

    if np.any(IQSample) > 0  and info.LambdaDivide is False:
        data_reduce.plot_data_with_errorbar([Q],[IQSample],[SigmaIQSample],label = [FileName+'Sample+Cell_stitched_'], title = FileName, xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath + '/' + 'StitchedDataSampleCell' + '/' + FileName + 'Sample+Cell_stitched.svg')    
    data_reduce.save_file4(Q,IQSample,SigmaIQSample,SigmaQ,info.OutPath +'/' + 'StitchedDataSampleCell' + '/' + SampleName + "Sample+Cell_Stitched.dat",ExperimentTimeSample)

    if info.igor_format == True:
        data_reduce.save_file4_igor(Q,IQSample,SigmaIQSample,SigmaQ,info.OutPath +'/' + 'StitchedDataSampleCell' + '/' + SampleName + "Sample+Cell_Stitched_igor.dat",ExperimentTimeSample)

    if np.any(IQ) > 0 and info.LambdaDivide is False:
        data_reduce.plot_data_with_errorbar([Q],[IQCell],[SigmaIQCell],label = [FileName+'OnlyCell_stitched_'], title = FileName, xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath + '/' + 'StitchedDataOnlyCell' + '/' + FileName + 'OnlyCell_stitched.svg')    
    data_reduce.save_file4(Q,IQCell,SigmaIQCell,SigmaQ,info.OutPath +'/' + 'StitchedDataOnlyCell' + '/' + SampleName + "OnlyCell_Stitched.dat",ExperimentTimeCell)
   
    if info.igor_format == True: 
        data_reduce.save_file4_igor(Q,IQCell,SigmaIQCell,SigmaQ,info.OutPath +'/' + 'StitchedDataOnlyCell' + '/' + SampleName + "OnlyCell_Stitched_igor.dat",ExperimentTimeCell)
#########################save data and figures
    
####################desmear data
    
    SigmaQ = de_none(SigmaQ)
    #print(SigmaQ)
    if info.DesmearData is True:
        Q,IQ,SigmaQ,n = extend_IQ(Q,IQ,SigmaQ)
        SigmaQ_max = np.max(SigmaQ)
        DeltaQ = np.linspace(-1*SigmaQ_max,SigmaQ_max,20)
        gauss_desmeared = data_reduce.vcg_gauss_desmear_array(Q,IQ,DeltaQ,SigmaQ)
        IQ[:n] = 0
        SigmaQ[:n] = 0
        gauss_desmeared[:n] = 0
        data_reduce.plot_data_with_errorbar([Q,Q],[IQ,gauss_desmeared],[SigmaIQ,SigmaIQ],label = [FileName+'OnlySample_stitched_',FileName+'OnlySample_desmeared'], title = FileName, xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True, logy = True, save = True,FileName = info.OutPath +'/' + 'StitchedDataOnlySample' +'/' + FileName + 'OnlySample_stitched_desmeared.svg')


        data_reduce.save_file4(Q,gauss_desmeared,SigmaIQ,SigmaQ,info.OutPath +'/' + 'StitchedDataOnlySample' + '/' + SampleName + "OnlySample_Stitched_desmeared.dat",ExperimentTimeSample)

####################desmear data
    
    
    
    
    
    
    
