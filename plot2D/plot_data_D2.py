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
        self.DataFold = DataFold # r'/data/hanzehua/vsanstrans'
        self.L1 = 12750       #mm
        self.L2 = 1400       #mm
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
        self.ModToDetector = 26000  #mm
        self.D21ToMod = 22000 + 4000
        self.D22ToMod = 22000 + 4300
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
        
        self.SDD21 = 4000
        self.SDD22 = 4430
        self.SDD23 = 4000
        self.SDD24 = 4430
        self.ThetaMin = np.arctan(300/self.SDD22)
        self.ThetaMax = np.arctan(700/self.SDD21)
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
    data1 = f["/csns/instrument/module21/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module22/histogram_data"][()] #got the right bank data
    data1 = np.array(np.sum(data1,axis = 0))
    data2 = np.array(np.sum(data2,axis = 0))

    tof21 = np.array(f["/csns/instrument/module21/time_of_flight"][()])
    tof22 = np.array(f["/csns/instrument/module22/time_of_flight"][()])
    #tof21 = tof21[data1>0]
    #tof22 = tof22[data2>0]
    tof21 = tof21/1000 # ms
    tof22 = tof22/1000 # ms
    WavelengthArray21 = info.const*tof21/info.D21ToMod
    WavelengthArray22 = info.const*tof22/info.D22ToMod    
    Qmin = 4*np.pi*np.sin(info.ThetaMin/2)/np.max(WavelengthArray22)
    Qmax = 4*np.pi*np.sin(info.ThetaMax/2)/np.min(WavelengthArray21)

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
    
    
def load_data_d2(RunNum,DataFold):
    DataInfo = instrument_info()
    tmp = DataInfo.WaveBins
#    if int(RunNum[-7:]) >= 4102:
#        TofPoints = 1000
#    else:
#        TofPoints = 5000
   # TofPoints = 500
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

    #data = f["/csns/instrument/module32/histogram_data"][()]    
    data1 = f["/csns/instrument/module21/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module22/histogram_data"][()] #got the right bank data
    data3 = f["/csns/instrument/module23/histogram_data"][()] #got the right bank data
    data4 = f["/csns/instrument/module24/histogram_data"][()] #got the right bank data
    #DataReshaped = np.reshape(data,(64,250,5000))
    #tmp = int(5000/DataInfo.TofBins)
    tof21 = f["/csns/instrument/module21/time_of_flight"][()]
    tof22 = f["/csns/instrument/module22/time_of_flight"][()]
 #   print(tof21,len(tof21))
 #   print(tof22,len(tof22))
    tofReshaped21 = np.average(np.reshape(tof21[:-1],(DataInfo.TofBins,tmp)),axis = 1)/1000 # ms
    tofReshaped22 = np.average(np.reshape(tof22[:-1],(DataInfo.TofBins,tmp)),axis = 1)/1000 # ms
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
    print('Total counts of \nD21: ' + str(Counts1) + '\nD22: ' + str(Counts2) + '\nD23: ' + str(Counts3) + '\nD24: ' + str(Counts4))

    print('Count rate of  (n/s)\nD21: ' + str(Counts1/useTime) + '\nD22: ' + str(Counts2/useTime) + '\nD23: ' + str(Counts3/useTime) + '\nD24: ' + str(Counts4/useTime))

    NormedCounts1, NormedCounts2, NormedCounts3, NormedCounts4 = np.sum(Data1Reshaped)/ProtonCharge*scale, np.sum(Data2Reshaped)/ProtonCharge*scale, np.sum(Data3Reshaped)/ProtonCharge*scale, np.sum(Data4Reshaped)/ProtonCharge*scale
    print('ProtonCharge normed counts of (n/PC*2.388E11) \nD21: ' + str(NormedCounts1) + '\nD22: ' + str(NormedCounts2) + '\nD23: ' + str(NormedCounts3) + '\nD24: ' + str(NormedCounts4))

    TNormedCounts1, TNormedCounts2, TNormedCounts3, TNormedCounts4 = np.sum(Data1Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data2Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data3Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data4Reshaped)/ProtonCharge*scale/useTimeMin
    print('Time and ProtonCharge normed counts of (n/min/PC*2.388E11) \nD21: ' + str(round(TNormedCounts1,2)) + '\nD22: ' + str(round(TNormedCounts2,2)) + '\nD23: ' + str(round(TNormedCounts3)) + '\nD24: ' + str(round(TNormedCounts4)))

    pc_t = ProtonCharge/useTimeMin
    print('ProtonCharge/Time/1E9 (PC/Min/1E9):\n',pc_t/1E9)

    return Data1Reshaped,Data2Reshaped,Data3Reshaped, Data4Reshaped,tofReshaped21,tofReshaped22




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

def plot_2d(D21,D22,D23,D24,save_name,show = True):
    info = instrument_info()
    TubeWidth = info.TubeWidth  #8.40625mm
    PixelHeight = info.PixelHeight   #mm
    tubes = info.tubes
    ShortPixels = info.ShortPixels
    LongPixels = info.LongPixels

    OriginD21 = [-300-TubeWidth*tubes,-500]
    OriginD22 = [-300,300]
    OriginD23 = [300,-500]
    OriginD24 = [-300,-300-TubeWidth*tubes]

    D21X = np.linspace(0,TubeWidth*tubes,tubes)
    D21Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D22X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D22Y = np.linspace(0,TubeWidth*tubes,tubes)
    D23X = np.linspace(0,TubeWidth*tubes,tubes)
    D23Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D24X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D24Y = np.linspace(0,TubeWidth*tubes,tubes)

    D21XP = D21X + OriginD21[0]
    D21YP = D21Y + OriginD21[1]
    D22XP = D22X + OriginD22[0]
    D22YP = D22Y + OriginD22[1]
    D23XP = D23X + OriginD23[0]
    D23YP = D23Y + OriginD23[1]
    D24XP = D24X + OriginD24[0]
    D24YP = D24Y + OriginD24[1]


    D21 = np.transpose(D21,(1,0,2))
    D23 = np.transpose(D23,(1,0,2))
    D22 = np.transpose(D22,(1,0,2))
    D24 = np.transpose(D24,(1,0,2))

    D21map =  np.sum(D21,axis = 2)
    with open(RunNum + '_D21_heat_map_data.txt','w') as f:
        for i in range(0,len(D21map)):
            for j in range(0,len(D21map[0])):
                print(D21map[i][j], file = f, end = '  ')
            print('        \n', file =f,end = '' )

    plt.figure(figsize=(5, 5))
    plt.contour(D21XP,D21YP,np.sum(D21,axis = 2),300,cmap = 'jet')
    plt.contour(D22XP,D22YP,np.sum(D22,axis = 2),300,cmap = 'jet')
    plt.contour(D23XP,D23YP,np.sum(D23,axis = 2),300,cmap = 'jet')
    plt.contour(D24XP,D24YP,np.sum(D24,axis = 2),300,cmap = 'jet')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(save_name + '_D2PositionPlot_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
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
    plt.title("2D map of " + save_name  + BankName  +' counts:'+ str(counts))
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
    if BankName == 'D21' or BankName == 'D23':
        WavelengthArray = datainfo.const*tof21/datainfo.D21ToMod
    elif BankName == 'D22' or BankName == 'D24':
        WavelengthArray = datainfo.const*tof22/datainfo.D22ToMod
    plt.plot(WavelengthArray,WavePlot)
    plt.title("Wavelength spectra of " + save_name  + BankName  +' counts:'+ str(counts))
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
    if BankName == 'D21' or BankName == 'D23':
        tof = tof21
    elif BankName == 'D22' or BankName == 'D24':
        tof = tof22
    TOFPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    plt.plot(tof,TOFPlot)
    plt.title("TOF of " + save_name  + BankName  +' counts:'+ str(counts))
    plt.xlabel('Time-of-Flight (ms)')
    plt.ylabel('Counts (n/ms)')
    plt.savefig(save_name+'_TOFPlot_'+ get_variable_name(DataStacked) +'.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
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
    OriginD21 = [-300-TubeWidth*tubes,-500]
    OriginD22 = [-300,300]
    OriginD23 = [300,-500]
    OriginD24 = [-300,-300-TubeWidth*tubes]

    D21X = np.linspace(0,TubeWidth*tubes,tubes)
    D21Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D22X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D22Y = np.linspace(0,TubeWidth*tubes,tubes)
    D23X = np.linspace(0,TubeWidth*tubes,tubes)
    D23Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D24X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D24Y = np.linspace(0,TubeWidth*tubes,tubes)

    D21XP = D21X + OriginD21[0]
    D21YP = D21Y + OriginD21[1]
    D22XP = D22X + OriginD22[0]
    D22YP = D22Y + OriginD22[1]
    D23XP = D23X + OriginD23[0]
    D23YP = D23Y + OriginD23[1]
    D24XP = D24X + OriginD24[0]
    D24YP = D24Y + OriginD24[1]
    #datainfo = instrument_info()
    counts = np.sum((DataStacked))
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    if BankName == 'D21':
        Y = D21YP
    elif BankName == 'D22':
        Y = D22YP
    elif BankName == 'D23':
        Y = D23YP
    elif BankName == 'D24':
        Y = D24YP
    YPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 1)
    plt.plot(Y,YPlot)
    plt.title("Vertical axis of " + save_name + BankName  +' counts:'+ str(counts))
    plt.xlabel('Vertical dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
    plt.savefig(save_name + '_YPlot_'+ get_variable_name(DataStacked) +'.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(Y,YPlot,save_name +'_YPlot_'+ get_variable_name(DataStacked) +'.dat')
    if show is True:
        plt.show()
    plt.close()


    if BankName == 'D21':
        X = D21XP 
    elif BankName == 'D22':
        X = D22XP
    elif BankName == 'D23': 
        X = D23XP 
    elif BankName == 'D24':
        X = D24XP
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

def plot_2d_q(D21,D22,D23,D24,save_name,Qmin,Qmax,show = True):
    datainfo = instrument_info()
    TubeWidth = ((15*0.5+8*15)*3+(2.5+8)*2)/48  #8.40625mm
    PixelHeight = 4   #mm
    tubes = 48
    ShortPixels = 150
    LongPixels = 250
    TofBins = 200
    OriginD21 = [-300-TubeWidth*tubes,-500]
    OriginD22 = [-300,300]
    OriginD23 = [300,-500]
    OriginD24 = [-300,-300-TubeWidth*tubes]

    D21X = np.linspace(0,TubeWidth*tubes,tubes)
    D21Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D22X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D22Y = np.linspace(0,TubeWidth*tubes,tubes)
    D23X = np.linspace(0,TubeWidth*tubes,tubes)
    D23Y = np.linspace(0,PixelHeight*LongPixels,LongPixels)
    D24X = np.linspace(0,PixelHeight*ShortPixels,ShortPixels)
    D24Y = np.linspace(0,TubeWidth*tubes,tubes)

    D21XP = D21X + OriginD21[0]
    D21YP = D21Y + OriginD21[1]
    D22XP = D22X + OriginD22[0]
    D22YP = D22Y + OriginD22[1]
    D23XP = D23X + OriginD23[0]
    D23YP = D23Y + OriginD23[1]
    D24XP = D24X + OriginD24[0]
    D24YP = D24Y + OriginD24[1]


    D21 = np.transpose(D21,(1,0,2))
    D23 = np.transpose(D23,(1,0,2))
    D22 = np.transpose(D22,(1,0,2))
    D24 = np.transpose(D24,(1,0,2))

    D21map =  np.sum(D21,axis = 2)
    with open(RunNum + '_D21_heat_map_data.txt','w') as f:
        for i in range(0,len(D21map)):
            for j in range(0,len(D21map[0])):
                print(D21map[i][j], file = f, end = '  ')
            print('        \n', file =f,end = '' )

    #plt.figure(figsize=(5, 5))
    #plt.contour(D21XP,D21YP,np.sum(D21,axis = 2),300,cmap = 'jet')
    #plt.contour(D22XP,D22YP,np.sum(D22,axis = 2),300,cmap = 'jet')
    #plt.contour(D23XP,D23YP,np.sum(D23,axis = 2),300,cmap = 'jet')
    #plt.contour(D24XP,D24YP,np.sum(D24,axis = 2),300,cmap = 'jet')
    #plt.xlabel('X (mm)')
    #plt.ylabel('Y (mm)')
    #plt.savefig(RunNum + '_D2PositionPlot_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    #plt.show()
    #plt.close()


    show = True #False
    #plot_data(D21,RunNum + 'D21',show);plot_data(D22,RunNum + 'D21',show);plot_data(D23,RunNum + 'D21',show);plot_data(D24,RunNum + 'D21',show)
    #plot_data(D22,RunNum + 'D21',show)

    info = instrument_info()
    D21ThetaX = np.arctan(D21XP/info.SDD21)
    D21ThetaY = np.arctan(D21YP/info.SDD21)
    D22ThetaX = np.arctan(D22XP/info.SDD22)
    D22ThetaY = np.arctan(D22YP/info.SDD22)
    D23ThetaX = np.arctan(D23XP/info.SDD23)
    D23ThetaY = np.arctan(D23YP/info.SDD23)
    D24ThetaX = np.arctan(D24XP/info.SDD24)
    D24ThetaY = np.arctan(D24YP/info.SDD24)

    D21QX = 4*np.pi*np.sin(D21ThetaX[:,None]/2)/info.WavelengthArray
    D21QY = 4*np.pi*np.sin(D21ThetaY[:,None]/2)/info.WavelengthArray
    D22QX = 4*np.pi*np.sin(D22ThetaX[:,None]/2)/info.WavelengthArray
    D22QY = 4*np.pi*np.sin(D22ThetaY[:,None]/2)/info.WavelengthArray
    D23QX = 4*np.pi*np.sin(D23ThetaX[:,None]/2)/info.WavelengthArray
    D23QY = 4*np.pi*np.sin(D23ThetaY[:,None]/2)/info.WavelengthArray
    D24QX = 4*np.pi*np.sin(D24ThetaX[:,None]/2)/info.WavelengthArray
    D24QY = 4*np.pi*np.sin(D24ThetaY[:,None]/2)/info.WavelengthArray
    
    QX = np.linspace(Qmin,Qmax,info.QBins)
    QY = np.linspace(Qmin,Qmax,info.QBins)
    QArray = np.zeros(len(QX))[:,None]*np.zeros(len(QY))
    # QX = info.QX
    # QY = info.QY
    # QArray = info.QArray

    QDict = [{},{},{},{}]

    for t in range(1,5):
        D2QXTmp = eval('D2' + str(t) + 'QX')
        for i in range(len(D2QXTmp)):
            D2QYTmp = eval('D2' + str(t) + 'QY')
            for j in range(len(D2QYTmp)):
                for k in range(len(info.WavelengthArray)):
                    Tmp1 = info.QBins - len(QX[QX > D2QXTmp[i,k]])-1
                    Tmp2 = info.QBins - len(QY[QY > D2QYTmp[j,k]])-1
                    QArray[Tmp2,Tmp1] += eval('D2' + str(t))[j,i,k]
    '''
                    if (Tmp2,Tmp1) in QDict[t-1].keys():
                        QDict[t-1][(Tmp2,Tmp1)].append([i,j,k])
                    else:
                        QDict[t-1][(Tmp2,Tmp1)] = []
                        QDict[t-1][(Tmp2,Tmp1)].append([i,j,k])



    QDictLoaded = []
    for i in range(0,4):
        np.save('QDictFile_D2' + str(i+1) + '.npy',QDict[i])
        #DataStacked = locals()['Data'+str(i)+'Reshaped']
        QDictLoaded.append(np.load('QDictFile_D2' + str(i+1) + '.npy', allow_pickle=True).item())

    for t in range(1,2):
        for key, items in QDictLoaded[t-1].items():
            #print(QDictLoaded)
            #print(t)
            tmp = np.array(items).copy()
            items = tmp
            if t == 1 or t == 3:
                QArray[key[0],key[1]] = np.sum(locals()['D2' + str(i)][items[:,1],items[:,0],items[:,2]])
            elif t == 2 or t == 4:
                QArray[key[0],key[1]] = np.sum(locals()['D2' + str(i)][items[:,0],items[:,1],items[:,2]])
    '''
    plt.figure(figsize=(5, 5))
    plt.contour(QX,QY,QArray,120,cmap = 'hot')
    plt.xlabel('QX (Angstrom$^{-1}$)')
    plt.ylabel('QY (Angstrom$^{-1}$)')
    plt.savefig(save_name + '_Q_XY_Plot_D2.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    if show == True:
        plt.show()
    plt.close()





#show = True # False
#plot_data(D21,RunNum + 'D21',show);plot_data(D22,RunNum + 'D21',show);plot_data(D23,RunNum + 'D21',show);plot_data(D24,RunNum + 'D21',show)




#DataFold = r'/data/hanzehua/vsanstrans'
RunNum = sys.argv[1]   

if RunNum[:3] == "RUN":
    pass
else:
    RunNum = r"RUN" + str('0'*(7-len(RunNum))) + RunNum 

print(RunNum) 

info = instrument_info()

QMinMax = get_q_min_max(RunNum)
Qmin = -1*QMinMax/2 #0.06       #A^-1
Qmax = QMinMax/2        #A^-1 
   

D21,D22,D23,D24,tof21,tof22 = load_data_d2(RunNum,DataFold)

show = True

if len(sys.argv) == 4:
    Cpara= sys.argv[3]
    if sys.argv[2] == get_variable_name(D21):
        Data3D = D21
    elif sys.argv[2] == get_variable_name(D22):
        Data3D = D22
    elif sys.argv[2] == get_variable_name(D23):
        Data3D = D23
    elif sys.argv[2] == get_variable_name(D24):
        Data3D = D24
    # Data3D = sys.argv[2]
    Data3D2 = get_variable_name(Data3D)
    
    if Cpara == '2D':
        plot_2d_sub(Data3D,RunNum + Data3D2, show)
    
    elif Cpara == 'lambda':
        plot_lambda(Data3D,RunNum + Data3D2, show)
    
    elif Cpara == 'tof':
        plot_tof(Data3D,RunNum + Data3D2, show)
    
    elif Cpara == 'xy':
        plot_xy(Data3D,RunNum + Data3D2, show)
    
    elif Cpara == '2DQ':
        plot_2d_q(D21,D22,D23,D24,RunNum + Data3D2,Qmin,Qmax, show)
        
elif len(sys.argv) == 3:
    Cpara= sys.argv[2]
    D21t = get_variable_name(D21)
    D22t = get_variable_name(D22)
    if Cpara == '2D':
        plot_2d(D21,D22,D23,D24,RunNum, show)
        
    elif Cpara == 'lambda':
        plot_lambda(D21,RunNum + D21t, show)
        plot_lambda(D22,RunNum + D22t, show)
        
    elif Cpara == 'tof':
        plot_tof(D21,RunNum + D21t, show)
        plot_tof(D22,RunNum + D22t, show)
    
    elif Cpara == 'xy':
        plot_xy(D21,RunNum + D21t, show)
        plot_xy(D22,RunNum + D22t, show)
    
    elif Cpara == '2DQ':
        plot_2d_q(D21,D22,D23,D24,RunNum,Qmin,Qmax, show)
        
else:
    D21t = get_variable_name(D21)
    D22t = get_variable_name(D22)    
    plot_2d(D21,D22,D23,D24,RunNum, show)

    plot_lambda(D21,RunNum + D21t, show)
    plot_lambda(D22,RunNum + D22t, show)

    plot_tof(D21,RunNum + D21t, show)
    plot_tof(D22,RunNum + D22t, show)

    plot_xy(D21,RunNum + D21t, show)
    plot_xy(D22,RunNum + D22t, show)

    plot_2d_q(D21,D22,D23,D24,RunNum + D21t,Qmin,Qmax, show)
  
    









