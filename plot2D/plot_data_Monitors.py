# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:11:46 2023

@author: zuots
"""
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
#directory = os.path.abspath(os.path.join(os.getcwd(), "../modules"))
#sys.path.append(directory)
#import instrument_reader as info
#print('info = ' + str(info.instrument_info))
class data_info():
    def __init__(self):
        self.L1 = 12750       #mm
        self.L2 = 11500       #mm
        self.A1 = 30          #mm
        self.A2 = 8           #mm
        self.BankWidth = 1094  #mm
        self.BankHeight = 1000  #mm
        self.TubeHeight = 4 #mm
        self.TubeWidth = 8.546875   #mm
        self.const = 3956.2   
        self.ModToDetector = 33500  #mm
        self.D3ToMod = 33500
        self.M3ToMod = 19988
        self.M6ToMod = 22360
        self.TimeDelay = 17.628 #info.instrument_info.TimeDelayD3 #17.628#52.7     #18.628  #ms  6埃，2.2埃和4埃的延时分别为：52.7ms，19.32ms和35.135ms
        self.TOF = 40              #ms
        #self.TofBins = 100
        self.QMin = 0.002        #A^-1
        self.QMax = 0.13 # 0.13 #0.05          #A^-1
        self.QBins = 120   
        self.WaveBins = 250 #5000
        self.TofBins = self.WaveBins
        self.XBins = 128
        self.YBins = 250
        self.RBins = 200
        self.R =900      #mm
        self.DeltaLambdaRatio = 0.019*self.const/self.ModToDetector
        self.XCenter = 65.9*self.TubeWidth -self.BankWidth/2    # mm
        self.YCenter = 124.37*self.TubeHeight - self.BankHeight/2     # mm
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
        self.XArray = np.arange(-self.TubeWidth*self.XBins/2,self.TubeWidth*self.XBins/2,self.TubeWidth)
        self.QArray = np.zeros(len(self.QX))[:,None]*np.zeros(len(self.QY))

    def save_file2(self,xx,yy,file_name):
        with open(file_name,'w') as f:
            for x,y in zip(xx,yy):
                print('{:<20.8f}{:>20.8f}'.format(x,y),file = f)
        f.close()


loc = locals()
def get_variable_name(variable):
    for k,v in loc.items():
        if loc[k] is variable:
            return k

def WavelengthPlot(WavelengthArray,Data, save_name ,show): 
    #WavePlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    plt.plot(WavelengthArray, Data)
    plt.xlabel('Neutron wavelegth (Angstrom)')
    plt.ylabel('Counts (n/A)')
    plt.title("Wavelength spectra of ")# + get_variable_name(Data))
    plt.savefig(save_name + '_Lambda_.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    #data_info.save_file2(WavelengthArray,Data, save_name +'_WavelengthPlot_' + get_variable_name(Data) + '.dat')
    if show == True:
        plt.show()
    plt.close()

def TransPlot(WavelengthArray,Data, save_name ,show):
    global datainfo
    #WavePlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    plt.plot(WavelengthArray, Data)
    plt.xlabel('Neutron wavelegth (Angstrom)')
    plt.ylabel('Transmission')
    #plt.title("Transmission of " + get_variable_name(Data))
    plt.savefig(save_name + '_Lambda_.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    datainfo.save_file2(WavelengthArray,Data, save_name +'_TransmissionPlot_'  + '.dat')
    if show == True:
        plt.show()
    plt.close()


def zero_divide(a,b):
    c = np.divide(a,b,out = np.zeros_like(a), where = b!=0)
    return c

 
def plot_data(M3, M6, M3S, M6S, tof3, tof6, RunNum, show = True):  
    global datainfo
    datainfo = data_info()
 
    WavelengthArrayM3 = datainfo.const*tof3/datainfo.M3ToMod 
    WavelengthArrayM6 = datainfo.const*tof6/datainfo.M6ToMod
    M6Interp = np.interp(WavelengthArrayM3,WavelengthArrayM6,M6,left = 0, right = 0)
    M6SInterp = np.interp(WavelengthArrayM3,WavelengthArrayM6,M6S,left = 0, right = 0)
    WavelengthPlot(WavelengthArrayM3,M6Interp,RunNum,show)
    WavelengthPlot(WavelengthArrayM3,M6SInterp,RunNum,show)
    Trans = zero_divide(M6SInterp,M6Interp)
    TransPlot(WavelengthArrayM3,Trans,RunNum,show)
    


#loc = locals()
#def get_variable_name(variable):
#    for k,v in loc.items():
#        if loc[k] is variable:
#            return k



def load_data2(RunNum):
    datainfo = data_info()
    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
        
    f = h5py.File(DataFileName, "r")
    data1 = f["/csns/instrument/monitor03/histogram_data"][()] #
    data2 = f["/csns/instrument/monitor06/histogram_data"][()] 
#    data1 = f["/csns/histogram_data/module32/histogram_data"][()] #got the left bank data
#    data2 = f["/csns/histogram_data/module31/histogram_data"][()] #got the right bank data
    #print(data1)
    #print(data2)
    #DataReshaped = np.reshape(data,(64,250,5000)i)
    ProtonCharge = f["/csns/proton_charge"][()]
    tmp = datainfo.WaveBins
    if int(RunNum[-4:]) >= 4102:
        TofPoints = 500
    else:
        TofPoints = 5000
    tmp2 = int(TofPoints/tmp)    
    tof3 = f["/csns/instrument/monitor03/time_of_flight"][()]
    tof6 = f["/csns/instrument/monitor06/time_of_flight"][()]
    #print(len(tof3))
    tofReshaped3 = np.average(np.reshape(tof3[:-1],(datainfo.TofBins,tmp2)),axis = 1)/1000 
    tofReshaped6 = np.average(np.reshape(tof6[:-1],(datainfo.TofBins,tmp2)),axis = 1)/1000
    Data1Reshaped = np.reshape(data1,(tmp,tmp2))
    Data1Reshaped2 = np.sum(Data1Reshaped,axis = 1)*9000*390701/ProtonCharge
    Data2Reshaped = np.reshape(data2,(tmp,tmp2))
    Data2Reshaped2 = np.sum(Data2Reshaped,axis = 1)*9000*390701/ProtonCharge
#    Data1Reshaped = np.reshape(data1,(64,250,5000))
#    Data2Reshaped = np.reshape(data2,(64,250,5000))
    #DataStacked = np.vstack((Data2Reshaped2,Data1Reshaped2))
    #print(DataStacked[DataStacked<=0])
#    DataStacked[:,:,90:109] =0
#    DataStacked[:,:,0:10] =0       
    print('The run number is : ' + str(RunNum))
    print('The proton charge is: ' + str(ProtonCharge) + '\n')
    NormedCounts1, NormedCounts2 = np.sum(Data1Reshaped)/ProtonCharge*9000*390701, np.sum(Data2Reshaped)/ProtonCharge*9000*390701
    print('Total Normed counts of \nM3: ' + str(NormedCounts1) + '\nM6: ' + str(NormedCounts2) )

    Counts1, Counts2 = np.sum(Data1Reshaped)*9000, np.sum(Data2Reshaped)*9000
    print('Total counts of \nM3: ' + str(Counts1) + '\nM6: ' + str(Counts2) )
    #DataStacked[:,:,130:145] = 0
    #DataStacked[26:100,:,:] = 0
    return Data1Reshaped2, Data2Reshaped2, tofReshaped3, tofReshaped6


DataFold = r'/data/hanzehua/vsanstrans'
RunInfoFold = r'/data/zuotaisen/VSANS_data_reduce/RunInfo'

RunNum = sys.argv[1]
if len(sys.argv) >= 3:
    RunNum2 = sys.argv[2]
else:
     RunNum2 = sys.argv[1]
#RunNumBkg = sys.argv[2]
#RunNum = r'RUN0002427'  #r'RUN0001827'
print(RunNum)
print(RunNum2)
#print(RunNumBkg)
Info = data_info()
M3,M6,tof3,tof6 = load_data2(RunNum)
M3S,M6S,tof3S,tof6S = load_data2(RunNum2)


show = False
plot_data(M3, M6, M3S, M6S,tof3,tof6, RunNum2, show)
    


#plt.figure(figsize=(5, 5))
#Data3D_2 = np.log10(np.sum(DataD3,axis = 2))
#Data3D_2 = np.sum(np.reshape(DataD3,(128,125,2)),axis = 2)
#ax = plt.matshow(Data3D_2,cmap = plt.cm.jet)
#plt.ylabel('Vertical dimension(Pixels)')
#plt.xlabel('Horizontal diemension(Pixels)')
#plt.savefig('PositionPlotD3_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
#plt.show()
'''
D3ThetaX = np.arctan(D3XP/Info.L2)
D3ThetaY = np.arctan(D3YP/Info.L2)


D3QX = 4*np.pi*np.sin(D3ThetaX[:,None]/2)/Info.WavelengthArray
D3QY = 4*np.pi*np.sin(D3ThetaY[:,None]/2)/Info.WavelengthArray

QX = Info.QX
QY = Info.QY
QArray = Info.QArray
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

np.save('QDictFileD3.npy',QDict)

QDictLoaded = np.load('QDictFileD3.npy').item()

for key, items in QDictLoaded.items():
    #print(QDictLoaded)
    items = np.array(items)
    QArray[key[0],key[1]] = np.sum(Data3D[items[:,0],items[:,1],items[:,2]])


QArray_2 = np.log10(QArray)
plt.figure(figsize=(5, 5))
ax = plt.matshow(QArray_2,cmap = plt.cm.jet)
plt.xlabel('QX')
plt.ylabel('QY')
plt.savefig(RunNum + '_Q_Matrix_PlotD3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
if show == True:
    plt.show()
plt.close()

plt.figure(figsize=(5, 5))
plt.contour(QX,QY,QArray_2,500,cmap = 'hot')
plt.xlabel('QX (Angstrom$^{-1}$)')
plt.ylabel('QY (Angstrom$^{-1}$)')
plt.savefig(RunNum + '_Q_XY_Plot_D3.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
if show == True:
    plt.show()
plt.close()


'''









