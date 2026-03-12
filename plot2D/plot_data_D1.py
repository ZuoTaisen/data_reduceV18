# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 06:27:53 2023

@author: zuots
"""
import sys
from math import *
import numpy
import scipy
import scipy.optimize
from matplotlib import pyplot as plt

import numpy as np
import h5py
import re
import os
import glob
import scipy
import scipy.optimize
import imp
import inspect
from datetime import datetime
cpath = os.getcwd()
sys.path.append(cpath + '//' + r'../')
import data_dir
import time_func

DataFold = data_dir.DataFold

class instrument_info():
    def __init__(self):
        self.DataFold = DataFold #r'/data/hanzehua/vsanstrans'
        self.L1 = 12750       #mm
        self.L2 = 1000       #mm
        self.A1 = 30          #mm
        self.A2 = 8           #mm
        self.tubes = 48
        self.BankWidth = 1094  #mm
        self.BankHeight = 1000  #mm
        self.TubeHeight = 4 #mm
        self.TubeWidth = 8.546875   #mm
        self.PixelHeight = 4
        self.ShortPixels = 150
        self.LongPixels = 250
        self.const = 3956.2   
        self.ModToDetector = 23000  #mm
        self.D11ToMod = 22000 + 1000
        self.D12ToMod = 22000 + 1430
        self.TimeDelay = 18.628    #18.628  #ms  6埃，2.2埃和4埃的延时分别为：52.7ms，19.32ms和35.135ms
        self.TOF = 40              #ms
        # self.QMin = -0.3 #0.06       #A^-1
        # self.QMax = 0.3           #A^-1
    
        self.QBins = 240 
        

        self.TofBins = 250 #100
        self.WaveBins = self.TofBins
        self.XBins = 128
        self.YBins = 250
        self.RBins = 200
        self.R = 900      #mm
        
        self.SDD11 = 1000
        self.SDD12 = 1430
        self.SDD13 = 1000
        self.SDD14 = 1430
        self.ThetaMin = np.arctan(300/self.SDD12)
        self.ThetaMax = np.arctan(700/self.SDD11)
        #self.QArray = np.linspace(self.QMin,self.QMax,self.QBins)
        self.DeltaLambdaRatio = 0.019*self.const/self.ModToDetector
        self.XCenter = 65.9*self.TubeWidth -self.BankWidth/2    # mm
        self.YCenter = 124.37*self.TubeHeight - self.BankHeight/2     # mm
        self.SampleThickness = 1   #mm
        #self.QX = np.logspace(log10(self.QMin),log10(self.QMax),self.QBins)
        #self.QY = np.logspace(log10(self.QMin),log10(self.QMax),self.QBins)
        

        self.StartWavelength = self.const*self.TimeDelay/self.ModToDetector
        self.StopWavelength = self.const*(self.TimeDelay+self.TOF)/self.ModToDetector
        self.WaveBand = self.StopWavelength - self.StartWavelength
        self.WaveBin = self.WaveBand/self.WaveBins
        self.WavelengthArray = np.arange(self.StartWavelength+self.WaveBin/2,self.StopWavelength,self.WaveBin)
        self.XArray = np.arange(-self.TubeWidth*self.XBins/2,self.TubeWidth*self.XBins/2,self.TubeWidth)

    def save_file2(self,xx,yy,file_name):
        with open(file_name,'w') as f:
            for x,y in zip(xx,yy):
                print('{:<20.8f}{:>20.8f}'.format(x,y),file = f)
        f.close()
        
def get_q_min_max(RunNum):
    info = instrument_info()
    DataFold = info.DataFold
    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
    

    f = h5py.File(DataFileName, "r")    
    data1 = f["/csns/instrument/module11/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module12/histogram_data"][()] #got the right bank data
    data1 = np.array(np.sum(data1,axis = 0))
    data2 = np.array(np.sum(data2,axis = 0))

    tof21 = np.array(f["/csns/instrument/module11/time_of_flight"][()])
    tof22 = np.array(f["/csns/instrument/module12/time_of_flight"][()])
    #tof21 = tof21[data1>0]
    #tof22 = tof22[data2>0]
    tof21 = tof21/1000 # ms
    tof22 = tof22/1000 # ms



    WavelengthArray21 = info.const*tof21/info.D11ToMod
    WavelengthArray22 = info.const*tof22/info.D12ToMod    
    Qmin = 4*np.pi*np.sin(info.ThetaMin/2)/np.max(WavelengthArray21)
    Qmax = 4*np.pi*np.sin(info.ThetaMax/2)/np.min(WavelengthArray22)

    return Qmax/4 

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

    
def load_data_D1(RunNum,DataFold):
    DataInfo = instrument_info()
    tmp = DataInfo.WaveBins
#    if int(RunNum[-7:]) >= 4102:
#        TofPoints = 1000
#    else:
#        TofPoints = 5000
#    TofPoints = 500
#    tmp = int(TofPoints/tmp)

    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
    infoFileName = DataFold + "/" + str(RunNum) + "/" + str(RunNum)
    f = h5py.File(DataFileName, "r")
    try:
        freq_ratio = f["/csns/Freq_ratio"][()]
    except:
        freq_ratio = 1

    if int(RunNum[-7:]) >= 4102:
        TofPoints = 500*freq_ratio
    else:
        TofPoints = 5000
    tmp = int(TofPoints/tmp)
    #startTime,endTime,useTime,useTimeMin = time_func.get_experimental_time(DataFold,RunNum)
    startTime,endTime,useTime,useTimeMin = time_func.get_experimental_time_info(DataFold,RunNum)
    startPulse = get_start_pulse(infoFileName)
    endPulse = get_end_pulse(infoFileName)
    Pulses = endPulse - startPulse
    ProtonCharge = f["/csns/proton_charge"][()]

    #print(RunNum,':' , ProtonCharge,round(useTime[1],2),round(ProtonCharge/useTime/1E6,2))
    #data = f["/csns/instrument/module32/histogram_data"][()]    
    data1 = f["/csns/instrument/module11/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module12/histogram_data"][()] #got the right bank data
    data3 = f["/csns/instrument/module13/histogram_data"][()] #got the right bank data
    data4 = f["/csns/instrument/module14/histogram_data"][()] #got the right bank data
    #DataReshaped = np.reshape(data,(64,250,5000))
    #tmp = int(5000/DataInfo.TofBins)
    tof21 = f["/csns/instrument/module11/time_of_flight"][()]
    tof22 = f["/csns/instrument/module12/time_of_flight"][()]

 #   print(tof21,len(tof21))
 #   print(tof22,len(tof22))
    tofReshapeD11 = np.average(np.reshape(tof21[:-1],(DataInfo.TofBins,tmp)),axis = 1)/1000 # ms
    tofReshapeD12 = np.average(np.reshape(tof22[:-1],(DataInfo.TofBins,tmp)),axis = 1)/1000 # ms
    ProtonCharge = f["/csns/proton_charge"][()]
    Data1Reshaped = np.sum(np.reshape(data1,(48,250,DataInfo.TofBins,tmp)),axis = 3)
    Data2Reshaped = np.sum(np.reshape(data2,(48,150,DataInfo.TofBins,tmp)),axis = 3)
    Data3Reshaped = np.sum(np.reshape(data3,(48,250,DataInfo.TofBins,tmp)),axis = 3)
    Data4Reshaped = np.sum(np.reshape(data4,(48,150,DataInfo.TofBins,tmp)),axis = 3)
#    DataStacked = np.vstack((Data2Reshaped,Data1Reshaped))   
    Data2Reshaped = np.transpose(Data2Reshaped,(1,0,2))
    Data4Reshaped = np.transpose(Data4Reshaped,(1,0,2))
    scale = 2.388E11

    print('The run number is : ' + str(RunNum))
    print('Start time is:',startTime)
    print('End time is:',endTime)
    print('Used time is:',useTime,'seconds or ',np.round(useTimeMin,2),'Minutes')   

    print('The proton charge is : ' + str(ProtonCharge) + '\n')
    Counts1, Counts2, Counts3, Counts4 = np.sum(Data1Reshaped), np.sum(Data2Reshaped), np.sum(Data3Reshaped),np.sum(Data4Reshaped)    
    print('Total counts of \nD11: ' + str(round(Counts1)) + '\nD12: ' + str(round(Counts2)) + '\nD13: ' + str(round(Counts3)) + '\nD14: ' + str(round(Counts4)))

    print('Count rate of (n/s)\nD11: ' + str(Counts1/useTime) + '\nD12: ' + str(Counts2/useTime) + '\nD13: ' + str(Counts3/useTime) + '\nD14: ' + str(Counts4/useTime))


    NormedCounts1, NormedCounts2, NormedCounts3, NormedCounts4 = np.sum(Data1Reshaped)/ProtonCharge*scale, np.sum(Data2Reshaped)/ProtonCharge*scale, np.sum(Data3Reshaped)/ProtonCharge*scale, np.sum(Data4Reshaped)/ProtonCharge*scale
    print('Proton Charge normed counts of (n/PC*2.388E11) \nD11: ' + str(round(NormedCounts1)) + '\nD12: ' + str(round(NormedCounts2)) + '\nD13: ' + str(round(NormedCounts3)) + '\nD14: ' + str(round(NormedCounts4)))

    TNormedCounts1, TNormedCounts2, TNormedCounts3, TNormedCounts4 = np.sum(Data1Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data2Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data3Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data4Reshaped)/ProtonCharge*scale/useTimeMin
    print('Time and Proton Charge normed counts of (n/min/PC*2.388E11) \nD11: ' + str(round(TNormedCounts1,2)) + '\nD12: ' + str(round(TNormedCounts2,2)) + '\nD13: ' + str(round(TNormedCounts3)) + '\nD14: ' + str(round(TNormedCounts4)))

    pc_t = ProtonCharge/useTimeMin
    print('ProtonCharge/Time/1E9 (PC/Min/1E9):\n',pc_t/1E9)
    
    return Data1Reshaped,Data2Reshaped,Data3Reshaped, Data4Reshaped,tofReshapeD11,tofReshapeD12




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

def plot_2d(D11,D12,D13,D14,save_name,show = True):
    info = instrument_info()
    TubeWidth = info.TubeWidth  #8.40625mm
    PixelHeight = info.PixelHeight   #mm
    tubes = info.tubes
    ShortPixels = info.ShortPixels
    LongPixels = info.LongPixels

    OriginD11 = [-300-TubeWidth*tubes,-500]
    OriginD12 = [-300,300]
    OriginD13 = [300,-500]
    OriginD14 = [-300,-300-TubeWidth*tubes]

    D11X = np.linspace(0,TubeWidth*tubes,tubes)
    D11Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D12X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D12Y = np.linspace(0,TubeWidth*tubes,tubes)
    D13X = np.linspace(0,TubeWidth*tubes,tubes)
    D13Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D14X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D14Y = np.linspace(0,TubeWidth*tubes,tubes)

    D11XP = D11X + OriginD11[0]
    D11YP = D11Y + OriginD11[1]
    D12XP = D12X + OriginD12[0]
    D12YP = D12Y + OriginD12[1]
    D13XP = D13X + OriginD13[0]
    D13YP = D13Y + OriginD13[1]
    D14XP = D14X + OriginD14[0]
    D14YP = D14Y + OriginD14[1]


    D11 = np.transpose(D11,(1,0,2))
    D13 = np.transpose(D13,(1,0,2))
    D12 = np.transpose(D12,(1,0,2))
    D14 = np.transpose(D14,(1,0,2))

    D11map =  np.sum(D11,axis = 2)
    with open(RunNum + '_D11_heat_map_data.txt','w') as f:
        for i in range(0,len(D11map)):
            for j in range(0,len(D11map[0])):
                print(D11map[i][j], file = f, end = '  ')
            print('        \n', file =f,end = '' )

    plt.figure(figsize=(5, 5))
    plt.contour(D11XP,D11YP,np.sum(D11,axis = 2),300,cmap = 'jet')
    plt.contour(D12XP,D12YP,np.sum(D12,axis = 2),300,cmap = 'jet')
    plt.contour(D13XP,D13YP,np.sum(D13,axis = 2),300,cmap = 'jet')
    plt.contour(D14XP,D14YP,np.sum(D14,axis = 2),300,cmap = 'jet')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(save_name + '_D1PositionPlot_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()
    plt.close()



def plot_2d_sub(DataStacked,save_name,show = True):  
    datainfo = instrument_info()
    Data2D = np.log10(np.sum(DataStacked[:,:,:],axis = 2))
    counts = np.sum((DataStacked))
    ax = plt.matshow(np.transpose(Data2D),cmap = plt.cm.hsv)
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    #plt.colorbar(ax.colorbar,fraction = 1000)
    #plt.title("2D map of " + str(get_variable_name(DataStacked)))
    plt.colorbar(label='log10(Counts)')#.set_label(Data2D)
    plt.title("2D map of " + save_name + BankName  +' counts:'+ str(counts))
    plt.xlabel('Vertical dimension(Pixels)')
    plt.ylabel('Horizontal diemension(Pixels)')
    plt.savefig(save_name +'_Matrix_plot_'+ get_variable_name(DataStacked) + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    if show is True:
        plt.show()
    plt.close()

def plot_lambda(DataStacked,save_name,show = True):  
    datainfo = instrument_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    WavePlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    if BankName == 'D11' or BankName == 'D13':
        WavelengthArray = datainfo.const*tof21/datainfo.D11ToMod
    elif BankName == 'D12' or BankName == 'D14':
        WavelengthArray = datainfo.const*tof22/datainfo.D12ToMod
    plt.plot(WavelengthArray,WavePlot)
    plt.title("Wavelength spectra of " + save_name  + BankName +' counts:'+ str(counts))
    plt.xlabel('Neutron wavelegth (Angstrom)')
    plt.ylabel('Counts (n/A)')
    plt.savefig(save_name+'_WavelengthPlot_'+ get_variable_name(DataStacked) +'.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(WavelengthArray,WavePlot,save_name +'_WavelengthPlot_'+ get_variable_name(DataStacked) +'.dat')
    if show is True:
        plt.show()
    plt.close()

def plot_tof(DataStacked,save_name,show = True):  
    datainfo = instrument_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    if BankName == 'D11' or BankName == 'D13':
        tof = tof21
    elif BankName == 'D12' or BankName == 'D14':
        tof = tof22
    TOFPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    plt.plot(tof,TOFPlot)
    plt.title("TOF of " + save_name + BankName +' counts:'+ str(counts))
    plt.xlabel('Time-of-Flight (ms)')
    plt.ylabel('Counts (n/ms)')
    plt.savefig(save_name+'_TOFPlot_'+  +'.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(tof,TOFPlot,save_name +'_TOFData_'+ get_variable_name(DataStacked) +'.dat')
    if show is True:
        plt.show()
    plt.close()

def plot_xy(DataStacked,save_name,show = True):  
    datainfo = instrument_info()
    TubeWidth = ((15*0.5+8*15)*3+(2.5+8)*2)/48  #8.40625mm
    PixelHeight = 4   #mm
    tubes = 48
    ShortPixels = 150
    LongPixels = 250
    TofBins = 200
    OriginD11 = [-300-TubeWidth*tubes,-500]
    OriginD12 = [-300,300]
    OriginD13 = [300,-500]
    OriginD14 = [-300,-300-TubeWidth*tubes]

    D11X = np.linspace(0,TubeWidth*tubes,tubes)
    D11Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D12X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D12Y = np.linspace(0,TubeWidth*tubes,tubes)
    D13X = np.linspace(0,TubeWidth*tubes,tubes)
    D13Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D14X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D14Y = np.linspace(0,TubeWidth*tubes,tubes)

    D11XP = D11X + OriginD11[0]
    D11YP = D11Y + OriginD11[1]
    D12XP = D12X + OriginD12[0]
    D12YP = D12Y + OriginD12[1]
    D13XP = D13X + OriginD13[0]
    D13YP = D13Y + OriginD13[1]
    D14XP = D14X + OriginD14[0]
    D14YP = D14Y + OriginD14[1]

    datainfo = instrument_info()
    counts = np.sum((DataStacked))
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    if BankName == 'D11':
        Y = D11YP
    elif BankName == 'D12':
        Y = D12YP
    elif BankName == 'D13':
        Y = D13YP
    elif BankName == 'D14':
        Y = D14YP
    YPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 1)
    plt.plot(Y,YPlot)
    plt.title("Vertical axis of " + save_name  + BankName  +' counts:'+ str(counts))
    plt.xlabel('Vertical dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
    plt.savefig(save_name + '_YPlot_'+ get_variable_name(DataStacked) +'.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(Y,YPlot,save_name +'_YPlot_'+ get_variable_name(DataStacked) +'.dat')
    if show is True:
        plt.show()
    plt.close()


    if BankName == 'D11':
        X = D11XP 
    elif BankName == 'D12':
        X = D12XP
    elif BankName == 'D13': 
        X = D13XP 
    elif BankName == 'D14':
        X = D14XP
    XPlot = np.sum(np.sum(DataStacked,axis = 1),axis = 1)
    plt.plot(XPlot)
    plt.title("Horizontal axis of " + save_name  + BankName  +' counts:'+ str(counts))
    plt.xlabel('Horizontal dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
    plt.savefig(save_name+'_XPlot_'+ get_variable_name(DataStacked) +'.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(X,XPlot,save_name +'_XData_'+ get_variable_name(DataStacked) +'.dat')
    if show is True:
        plt.show()
    plt.close()    

def plot_2d_q(D11,D12,D13,D14,save_name,Qmin,Qmax,show = True):
    datainfo = instrument_info()
    TubeWidth = ((15*0.5+8*15)*3+(2.5+8)*2)/48  #8.40625mm
    PixelHeight = 4   #mm
    tubes = 48
    ShortPixels = 150
    LongPixels = 250
    TofBins = 200
    OriginD11 = [-300-TubeWidth*tubes,-500]
    OriginD12 = [-300,300]
    OriginD13 = [300,-500]
    OriginD14 = [-300,-300-TubeWidth*tubes]

    D11X = np.linspace(0,TubeWidth*tubes,tubes)
    D11Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D12X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D12Y = np.linspace(0,TubeWidth*tubes,tubes)
    D13X = np.linspace(0,TubeWidth*tubes,tubes)
    D13Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D14X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D14Y = np.linspace(0,TubeWidth*tubes,tubes)

    D11XP = D11X + OriginD11[0]
    D11YP = D11Y + OriginD11[1]
    D12XP = D12X + OriginD12[0]
    D12YP = D12Y + OriginD12[1]
    D13XP = D13X + OriginD13[0]
    D13YP = D13Y + OriginD13[1]
    D14XP = D14X + OriginD14[0]
    D14YP = D14Y + OriginD14[1]


    D11 = np.transpose(D11,(1,0,2))
    D13 = np.transpose(D13,(1,0,2))
    D12 = np.transpose(D12,(1,0,2))
    D14 = np.transpose(D14,(1,0,2))

    D11map =  np.sum(D11,axis = 2)
    with open(RunNum + '_D11_heat_map_data.txt','w') as f:
        for i in range(0,len(D11map)):
            for j in range(0,len(D11map[0])):
                print(D11map[i][j], file = f, end = '  ')
            print('        \n', file =f,end = '' )

    #plt.figure(figsize=(5, 5))
    #plt.contour(D11XP,D11YP,np.sum(D11,axis = 2),300,cmap = 'jet')
    #plt.contour(D12XP,D12YP,np.sum(D12,axis = 2),300,cmap = 'jet')
    #plt.contour(D13XP,D13YP,np.sum(D13,axis = 2),300,cmap = 'jet')
    #plt.contour(D14XP,D14YP,np.sum(D14,axis = 2),300,cmap = 'jet')
    #plt.xlabel('X (mm)')
    #plt.ylabel('Y (mm)')
    #plt.savefig(RunNum + '_D1PositionPlot_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    #plt.show()
    #plt.close()


    show = True #False
    #plot_data(D11,RunNum + 'D11',show);plot_data(D12,RunNum + 'D11',show);plot_data(D13,RunNum + 'D11',show);plot_data(D14,RunNum + 'D11',show)
    #plot_data(D12,RunNum + 'D11',show)

    info = instrument_info()
    D11ThetaX = np.arctan(D11XP/info.SDD11)
    D11ThetaY = np.arctan(D11YP/info.SDD11)
    D12ThetaX = np.arctan(D12XP/info.SDD12)
    D12ThetaY = np.arctan(D12YP/info.SDD12)
    D13ThetaX = np.arctan(D13XP/info.SDD13)
    D13ThetaY = np.arctan(D13YP/info.SDD13)
    D14ThetaX = np.arctan(D14XP/info.SDD14)
    D14ThetaY = np.arctan(D14YP/info.SDD14)

    D11QX = 4*np.pi*np.sin(D11ThetaX[:,None]/2)/info.WavelengthArray
    D11QY = 4*np.pi*np.sin(D11ThetaY[:,None]/2)/info.WavelengthArray
    D12QX = 4*np.pi*np.sin(D12ThetaX[:,None]/2)/info.WavelengthArray
    D12QY = 4*np.pi*np.sin(D12ThetaY[:,None]/2)/info.WavelengthArray
    D13QX = 4*np.pi*np.sin(D13ThetaX[:,None]/2)/info.WavelengthArray
    D13QY = 4*np.pi*np.sin(D13ThetaY[:,None]/2)/info.WavelengthArray
    D14QX = 4*np.pi*np.sin(D14ThetaX[:,None]/2)/info.WavelengthArray
    D14QY = 4*np.pi*np.sin(D14ThetaY[:,None]/2)/info.WavelengthArray
    
    QX = np.linspace(Qmin,Qmax,info.QBins)
    QY = np.linspace(Qmin,Qmax,info.QBins)
    QArray = np.zeros(len(QX))[:,None]*np.zeros(len(QY))
    # QX = info.QX
    # QY = info.QY
    # QArray = info.QArray

    QDict = [{},{},{},{}]

    for t in range(1,5):
        D1QXTmp = eval('D1' + str(t) + 'QX')
        for i in range(len(D1QXTmp)):
            D1QYTmp = eval('D1' + str(t) + 'QY')
            for j in range(len(D1QYTmp)):
                for k in range(len(info.WavelengthArray)):
                    Tmp1 = info.QBins - len(QX[QX > D1QXTmp[i,k]])-1
                    Tmp2 = info.QBins - len(QY[QY > D1QYTmp[j,k]])-1
                    QArray[Tmp2,Tmp1] += eval('D1' + str(t))[j,i,k]
    '''
                    if (Tmp2,Tmp1) in QDict[t-1].keys():
                        QDict[t-1][(Tmp2,Tmp1)].append([i,j,k])
                    else:
                        QDict[t-1][(Tmp2,Tmp1)] = []
                        QDict[t-1][(Tmp2,Tmp1)].append([i,j,k])



    QDictLoaded = []
    for i in range(0,4):
        np.save('QDictFile_D1' + str(i+1) + '.npy',QDict[i])
        #DataStacked = locals()['Data'+str(i)+'Reshaped']
        QDictLoaded.append(np.load('QDictFile_D1' + str(i+1) + '.npy', allow_pickle=True).item())

    for t in range(1,2):
        for key, items in QDictLoaded[t-1].items():
            #print(QDictLoaded)
            #print(t)
            tmp = np.array(items).copy()
            items = tmp
            if t == 1 or t == 3:
                QArray[key[0],key[1]] = np.sum(locals()['D1' + str(i)][items[:,1],items[:,0],items[:,2]])
            elif t == 2 or t == 4:
                QArray[key[0],key[1]] = np.sum(locals()['D1' + str(i)][items[:,0],items[:,1],items[:,2]])
    '''
    plt.figure(figsize=(5, 5))
    plt.contour(QX,QY,QArray,120,cmap = 'hot')
    plt.xlabel('QX (Angstrom$^{-1}$)')
    plt.ylabel('QY (Angstrom$^{-1}$)')
    plt.savefig(save_name + '_Q_XY_Plot_D1.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    if show == True:
        plt.show()
    plt.close()





#show = True # False
#plot_data(D11,RunNum + 'D11',show);plot_data(D12,RunNum + 'D11',show);plot_data(D13,RunNum + 'D11',show);plot_data(D14,RunNum + 'D11',show)




#DataFold = r'/data/hanzehua/vsanstrans'
RunNum = sys.argv[1]   

if RunNum[:3] == "RUN":
    pass
else:
    RunNum = r"RUN" + str('0'*(7-len(RunNum))) + RunNum 

print(RunNum) 

info = instrument_info()

QMinMax = get_q_min_max(RunNum)
Qmin = -1*QMinMax #0.06       #A^-1
Qmax = QMinMax          #A^-1 
   

D11,D12,D13,D14,tof21,tof22 = load_data_D1(RunNum,DataFold)


show = True

if len(sys.argv) == 4:
    Cpara= sys.argv[3]
    if sys.argv[2] == get_variable_name(D11):
        Data3D = D11
    elif sys.argv[2] == get_variable_name(D12):
        Data3D = D12
    elif sys.argv[2] == get_variable_name(D13):
        Data3D = D13
    elif sys.argv[2] == get_variable_name(D14):
        Data3D = D14
    # Data3D = sys.argv[2]
    Data3D1 = get_variable_name(Data3D)
    
    if Cpara == '2D':
        plot_2d_sub(Data3D,RunNum + Data3D1, show)
    
    elif Cpara == 'lambda':
        plot_lambda(Data3D,RunNum + Data3D1, show)
    
    elif Cpara == 'tof':
        plot_tof(Data3D,RunNum + Data3D1, show)
    
    elif Cpara == 'xy':
        plot_xy(Data3D,RunNum + Data3D1, show)
    
    elif Cpara == '2DQ':
        plot_2d_q(D11,D12,D13,D14,RunNum + Data3D1,Qmin,Qmax, show)
        
elif len(sys.argv) == 3:
    Cpara= sys.argv[2]
    D11t = get_variable_name(D11)
    D12t = get_variable_name(D12)
    if Cpara == '2D':
        plot_2d(D11,D12,D13,D14,RunNum, show)
        
    elif Cpara == 'lambda':
        plot_lambda(D11,RunNum + D11t, show)
        plot_lambda(D12,RunNum + D12t, show)
        
    elif Cpara == 'tof':
        plot_tof(D11,RunNum + D11t, show)
        plot_tof(D12,RunNum + D12t, show)
    
    elif Cpara == 'xy':
        plot_xy(D11,RunNum + D11t, show)
        plot_xy(D12,RunNum + D12t, show)
    
    elif Cpara == '2DQ':
        plot_2d_q(D11,D12,D13,D14,RunNum,Qmin,Qmax, show)
        
else:
    D11t = get_variable_name(D11)
    D12t = get_variable_name(D12)    
    plot_2d(D11,D12,D13,D14,RunNum, show)

    plot_lambda(D11,RunNum + D11t, show)
    plot_lambda(D12,RunNum + D12t, show)

    plot_tof(D11,RunNum + D11t, show)
    plot_tof(D12,RunNum + D12t, show)

    plot_xy(D11,RunNum + D11t, show)
    plot_xy(D12,RunNum + D12t, show)

    plot_2d_q(D11,D12,D13,D14,RunNum + D11t,Qmin,Qmax, show)
  
    









