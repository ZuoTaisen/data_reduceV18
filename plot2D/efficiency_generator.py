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
import pickle
import argparse
pickle.DEFAULT_PROTOCOL = 4

DataFold = data_dir.DataFold

#directory = os.path.abspath(os.path.join(os.getcwd(), "../modules"))
#sys.path.append(directory)

class data_info():
    def __init__(self):
        self.DataFold = DataFold #r'/data/hanzehua/vsanstrans'
        self.L1 = 12750       #mm
        self.L2 = 11500       #mm
        self.A1 = 30          #mm
        self.A2 = 8           #mm
        #self.BankWidth = 1096  #mm
        self.BankHeight = 1000  #mm
        self.TubeHeight = 4 #mm
        self.TubeWidth = ((127.5)*4+10.5*3)/63   #mm   8.59523 
        self.BankGap = 0   #mm
        self.BankWidth = self.TubeWidth*63*2 + self.BankGap
        self.const = 3956.2   
        self.ModToDetector = 33500  #mm
        self.D3ToMod = 33500
        self.TimeDelay = 17.628 #info.instrument_info.TimeDelayD3 #17.628#52.7     #18.628  #ms  6埃，2.2埃和4埃的延时分别为：52.7ms，19.32ms和35.135ms
        self.TOF = 40              #ms
        #self.TofBins = 100
        self.QMin = 0.002        #A^-1
        self.QMax =  0.13 # 0.13 #0.05          #A^-1
        self.QBins = 120   
        self.WaveBins = 250 #250 #250 #5000
        self.TofBins = self.WaveBins
        self.XBins = 128
        self.YBins = 250
        self.RBins = 200
        self.R =900      #mm
        self.ThetaMin = 0
        self.ThetaMax = np.arctan(500*np.sqrt(2)/self.L2)
        self.DeltaLambdaRatio = 0.019*self.const/self.ModToDetector
        self.XCenter = 66.4*self.TubeWidth - self.TubeWidth*64 + self.BankGap/2 # 65.9*self.TubeWidth - self.TubeWidth*64 + self.BankGap/2    # mm
        self.YCenter = 124.9*self.TubeHeight - self.BankHeight/2 # 124.37*self.TubeHeight - self.BankHeight/2     # mm
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
        self.XArrayLeft = np.arange(-1*self.TubeWidth*self.XBins/2, self.TubeWidth, self.TubeWidth) - self.BankGap/2
        self.XArrayRight = np.arange(self.TubeWidth, self.TubeWidth*self.XBins/2, self.TubeWidth) + self.BankGap/2
        self.XArray = np.concatenate((self.XArrayLeft,self.XArrayRight)) - self.XCenter
        self.YArray = np.arange(-1*self.TubeHeight*self.YBins/2+self.TubeHeight/2,self.TubeHeight*self.YBins/2+self.TubeHeight/2,self.TubeHeight) + self.YCenter
        #self.XArray = np.arange(-self.TubeWidth*self.XBins/2,self.TubeWidth*self.XBins/2,self.TubeWidth)
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
    data1 = f["/csns/instrument/module31/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module32/histogram_data"][()] #got the right bank data
    data1 = np.array(np.sum(data1,axis = 0))
    data2 = np.array(np.sum(data2,axis = 0))

    tof31 = np.array(f["/csns/instrument/module31/time_of_flight"][()])
    tof32 = np.array(f["/csns/instrument/module32/time_of_flight"][()])
    #print(data1)
    #print(data2)
    #tof31 = tof31[data1>0]
    #tof32 = tof32[data2>0]
    tof31 = tof31/1000 # ms
    tof32 = tof32/1000 # ms
    #print(tof31)
    WavelengthArray31 = info.const*tof31/info.ModToDetector
    WavelengthArray32 = info.const*tof32/info.ModToDetector 
    #print(WavelengthArray32)
    Qmin = 4*np.pi*np.sin(info.ThetaMin/2)/np.max(WavelengthArray31)
    Qmax = 4*np.pi*np.sin(info.ThetaMax/2)/np.min(WavelengthArray32)
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
    data1 = f["/csns/instrument/module32/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module31/histogram_data"][()] #got the right bank data


    startTime = f["/csns/start_time_utc"][()][0]
    endTime = f["/csns/end_time_utc"][()][0]
    useTime = time_diff(startTime,endTime)
    useTimeMin = useTime[1]
    startPulse = get_start_pulse(infoFileName)
    endPulse = get_end_pulse(infoFileName)
    Pulses = endPulse - startPulse
    ProtonCharge = f["/csns/proton_charge"][()]
    try:
        freq_ratio = f["/csns/Freq_ratio"][()]
    except:
        freq_ratio = 1
    #freq_ratio = 2 #
    print(freq_ratio)
    tmp = datainfo.WaveBins
    if int(RunNum[-7:]) >= 4102:
        TofPoints = 500*freq_ratio
    else:
        TofPoints = 5000
    tmp2 = int(TofPoints/tmp)
  
    tof3 = f["/csns/instrument/module31/time_of_flight"][()]
    
    print(data1.shape)
    print(data2.shape)
    print(tof3.shape)
    tofReshaped3 = np.average(np.reshape(tof3[:-1],(datainfo.TofBins,tmp2)),axis = 1)/1000 
    Data1Reshaped = np.reshape(data1,(64,250,tmp,tmp2))
    Data1Reshaped2 = np.sum(Data1Reshaped,axis = 3)
    Data2Reshaped = np.reshape(data2,(64,250,tmp,tmp2))
    Data2Reshaped2 = np.sum(Data2Reshaped,axis = 3) 
#    Data1Reshaped = np.reshape(data1,(64,250,5000))
#    Data2Reshaped = np.reshape(data2,(64,250,5000))
    DataStacked = np.vstack((Data2Reshaped2,Data1Reshaped2))
    #DataStacked[0:80] = 0
    scale = 2.388E11
    
    print('The run number is : ' + str(RunNum))
    print('The run number is : ' + str(RunNum))
    print('Start time is:',startTime)
    print('End time is:',endTime)
    print('Used time is:',useTime[0],'seconds or ',np.round(useTime[1],2),'Minutes')   
    print('The proton charge is: ' + str(ProtonCharge) + '\n')
    
    Counts1, Counts2 = np.sum(Data1Reshaped), np.sum(Data2Reshaped)
    print('Total counts of \nD31: ' + str(round(Counts1)) + '\nD32: ' + str(round(Counts2)) )
    
    print('Count rate of  (n/s) \nD31: ' + str(round(Counts1/useTime[0])) + '\nD32: ' + str(round(Counts2/useTime[0])) )

    
    NormedCounts1, NormedCounts2 = np.sum(Data1Reshaped)/ProtonCharge*scale, np.sum(Data2Reshaped)/ProtonCharge*scale
    print('Proton Charge Normed counts of (n/PC*2.388E11) \nD31: ' + str(round(NormedCounts1)) + '\nD32: ' + str(round(NormedCounts2)) )

    TNormedCounts1, TNormedCounts2 = np.sum(Data1Reshaped)/ProtonCharge*scale/useTimeMin, np.sum(Data2Reshaped)/ProtonCharge*scale/useTimeMin
    print('Time and Proton Charge Normed counts of (n/min/PC*2.388E11) \nD31: ' + str(round(TNormedCounts1)) + '\nD32: ' + str(round(TNormedCounts2)) )

    pc_t = ProtonCharge/useTimeMin
    print('ProtonCharge/Time/1E9 (PC/Min/1E9):\n',pc_t/1E9)
    #DataStacked[:,:,10:] = 0 
    return DataStacked/ProtonCharge*scale,tofReshaped3
    
def plot_2d(DataStacked,save_name, show = True):  
    datainfo = data_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    Data2D = np.log10(np.sum(DataStacked[:,:,:],axis = 2))
    counts = np.sum((DataStacked))
    plt.figure(1)
    ax = plt.matshow(np.transpose(Data2D),cmap = plt.cm.jet)
    plt.title("2D map of " + save_name  + BankName +' counts:'+ str(counts))
    plt.colorbar(label='log10(Counts)')#.set_label(Data2D)
    plt.ylabel('Vertical dimension(Pixels)')
    plt.xlabel('Horizontal diemension(Pixels)')
    plt.savefig(save_name + '_MatrixPlot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)

    
    Data2D_2 = (np.sum(np.reshape(Data2D,(128,125,2)),axis = 2))
    plt.figure(2)
    ax = plt.matshow(np.transpose(Data2D_2),cmap = plt.cm.jet)
    #plt.colorbar(ax.colorbar,fraction = 1000)
    plt.title("2D map of " + save_name  + BankName +' counts:'+ str(counts))
    plt.colorbar(label='log10(Counts)')#.set_label(Data2D)
    plt.ylabel('Vertical dimension(Pixels)')
    plt.xlabel('Horizontal diemension(Pixels)')
    plt.savefig(save_name + '_MatrixPlot_D3_2.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    if show == True:
        plt.show()
    plt.close()

def plot_lambda(DataStacked,save_name, show = True):  
    datainfo = data_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    WavelengthArray = datainfo.const*tof3/datainfo.D3ToMod 
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

def plot_xy(DataStacked,save_name, show = True):  
    datainfo = data_info()
    BankNames = get_variable_names(DataStacked)
    BankName = min(BankNames, key=len)
    counts = np.sum((DataStacked))
    D3XP = datainfo.XArray
    D3YP = datainfo.YArray
    YPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 1)
    plt.figure(1)
    plt.plot(D3YP, YPlot)
    plt.title("Vertical axis of " + save_name  + BankName +' counts:'+ str(counts))
    plt.xlabel('Vertical dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
    plt.savefig(save_name+'_YPlot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(D3YP,YPlot,save_name +'_YData_D3'+ '.dat')

    plt.figure(2)    
    XPlot = np.sum(np.sum(DataStacked,axis = 1),axis = 1)
    plt.plot(D3XP, XPlot)
    plt.title("Horizontal axis of " + save_name  + BankName +' counts:'+ str(counts))
    plt.xlabel('Horizontal dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
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
    
    
    D3QX = 4*np.pi*np.sin(D3ThetaX[:,None]/2)/Info.WavelengthArray
    D3QY = 4*np.pi*np.sin(D3ThetaY[:,None]/2)/Info.WavelengthArray
    QX = np.linspace(Qmin,Qmax,Info.QBins)
    QY = np.linspace(Qmin,Qmax,Info.QBins)
    QArray = np.zeros(len(QX))[:,None]*np.zeros(len(QY))
#    QX = Info.QX
#    QY = Info.QY
#    QArray = Info.QArray
    QDict = {}
    
    for i in range(len(Data3D)):
        for j in range(len(Data3D[i])):
            for k in range(len(Info.WavelengthArray)):
                Tmp1 = Info.QBins - len(QX[QX > D3QX[i,k]])-1
                Tmp2 = Info.QBins - len(QY[QY > D3QY[j,k]])-1
               # QArray[Tmp2,Tmp1] += Data3D[i,j,k]
                if (Tmp2,Tmp1) in QDict.keys():
                    QDict[(Tmp2,Tmp1)].append([i,j,k])
                else:
                    QDict[(Tmp2,Tmp1)] = []
                    QDict[(Tmp2,Tmp1)].append([i,j,k])
    
    
    #if os.path.exists(r'QDictFileD3' + str(self.WaveMin) + '-' + str(self.WaveMax) + 'A.npy'):
    #np.save('QDictFileD3.pkl',QDict)
    with open('QDictFileD3.pkl','wb') as f:
        pickle.dump(QDict, f)
    with open('QDictFileD3.pkl','rb') as f:
        QDictLoaded = pickle.load(f)
    #QDictLoaded = np.load('QDictFileD3.pkl').item()
    
    for key, items in QDictLoaded.items():
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

def normalize_matrix(matrix):
    non_zero = [x for row in matrix for x in row if x != 0]
    avg = sum(non_zero) / len(non_zero) if non_zero else 1
    return [
        [1.0 if x == 0 else x/avg for x in row]
        for row in matrix
    ]
def save_matrix(matrix, filename):
    """保存矩阵到二进制文件"""
    np.save(filename, matrix)
    print(f"矩阵已保存到 {filename}.npy")

def load_matrix(filename):
    """从二进制文件加载矩阵"""
    matrix = np.load(f"{filename}.npy")
    return matrix
#DataFold = r'/data/hanzehua/vsanstrans'


RunNum = sys.argv[1]

if RunNum[:3] == "RUN":
    pass
else:
    RunNum = r"RUN" + str('0'*(7-len(RunNum.split('_')[0]))) + RunNum
    #RunNum = r"RUN" + str('0'*(7-len(RunNum))) + RunNum 
print(RunNum)

QMinMax = get_q_min_max(RunNum)
Qmin = -1*QMinMax*0.3 #0.06       #A^-1
Qmax = QMinMax*0.3          #A^-1  2.2-6.7A 取0.15   6-10.5埃取0.05

Info = data_info()
Data3D,tof3 = load_data2(RunNum)
D3 = Data3D
D3xy = np.sum(D3,axis = 2)
D3xy[0:128,50:200] = 0
#print(D3xy)
normMatrix = normalize_matrix(D3xy)
#for i in range(len(normMatrix)):
#    for j in range(len(normMatrix[0])):
#        if normMatrix[i][j] < 0.5:
#            print(normMatrix[i][j])
normMatrix2 = np.array(normMatrix)
#save_matrix(normMatrix2, 'normMatrix')
np.save(r'./normMatrix.npy', normMatrix2)
#np.save(r'./norm_matrix'+'.npy','wb',normMatrix2)
#with open('norm_matirx.pkl', 'wb') as f:
#    pickle.dump(normMatrix2, f)

#for i in range(len(normMatrix)):
#    plt.plot(normMatrix[i])
#plt.show()
loaded_matrix = np.load(r'./normMatrix.npy') #load_matrix('normMatrix')
print(loaded_matrix.shape)
