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
from numpy.polynomial import polynomial as P

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
        self.TimeDelay = 17.628 #info.instrument_info.TimeDelayD3 #17.628#52.7     #18.628  #ms  6埃，2.2埃和4埃的延时分别为：52.7ms，19.32ms和35.135ms
        self.TOF = 40              #ms
        #self.TofBins = 100
        self.QMin = 0.002        #A^-1
        self.QMax = 0.13#0.05          #A^-1
        self.QBins = 120   
        self.WaveBins = 125 #5000
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

    
def plot_data(DataStacked,save_name):  
    datainfo = data_info()
    Data2D = np.log10(np.sum(DataStacked[:,:,:],axis = 2))
    counts = np.sum(np.round(DataStacked))
    #ax = plt.matshow(np.transpose(Data2D),cmap = plt.cm.jet)
    #plt.title("2D map of " + get_variable_name(DataStacked) +' counts:'+ str(counts))
    #plt.colorbar()#.set_label(Data2D)
    #plt.ylabel('Vertical dimension(Pixels)')
    #plt.xlabel('Horizontal diemension(Pixels)')
    #plt.savefig(save_name+'_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    #plt.show()
    
    Data2D_2 = (np.sum(np.reshape(Data2D,(128,125,2)),axis = 2))
    ax = plt.matshow(np.transpose(Data2D_2),cmap = plt.cm.jet)
    #plt.colorbar(ax.colorbar,fraction = 1000)
    plt.title("2D map of " + get_variable_name(DataStacked) +' counts:'+ str(counts))
    plt.colorbar()#.set_label(Data2D)
    plt.ylabel('Vertical dimension(Pixels)')
    plt.xlabel('Horizontal diemension(Pixels)')
    plt.savefig(save_name+'_2D_2.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()


    
    #plt.figure(figsize=(5, 5))
#    for t in range(1,5):
#        plt.plot(Q[:-5],eval('QArrayD2'+str(t))[:-5],label = str(t))
#        plt.legend()
#    plt.xlabel('Q (Angstrom$^{-1}$)')
#    plt.ylabel('Counts')
#    #plt.savefig('QPlot_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
#    plt.show()
    WavelengthArray = datainfo.const*tof3/datainfo.D3ToMod 
    WavePlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    #plt.plot(datainfo.WavelengthArray,WavePlot)
    with open(save_name + 'lambda.dat','w') as f:
        for i in range(len(WavePlot)):
            print(WavelengthArray[i],'  ',WavePlot[i],file = f)
    plt.plot(WavelengthArray, WavePlot)
    plt.xlabel('Neutron wavelegth (Angstrom)')
    plt.ylabel('Counts (n/A)')
    plt.title("Wavelength spectra of " + get_variable_name(DataStacked) +' counts:'+ str(counts))
    plt.savefig(save_name+'_Lambda.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()
    datainfo.save_file2(datainfo.WavelengthArray,WavePlot,save_name +'wavelengthPlot.dat')


    TOFPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
    plt.plot(tof3,TOFPlot)
    plt.title("TOF of " + get_variable_name(DataStacked) +' counts:'+ str(counts))
    plt.xlabel('Time-of-Flight (ms)')
    plt.ylabel('Counts (n/ms)')
    plt.savefig(save_name+'_TOF.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()
    
    YPlot = np.sum(np.sum(DataStacked,axis = 0),axis = 1)
    plt.plot(YPlot)
    plt.title("Vertical axis of " + get_variable_name(DataStacked) +' counts:'+ str(counts))
    plt.xlabel('Vertical dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
    plt.savefig(save_name+'_YPlot.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()
    
    XPlot = np.sum(np.sum(DataStacked,axis = 1),axis = 1)
    plt.plot(XPlot)
    plt.title("Horizontal axis of " + get_variable_name(DataStacked) +' counts:'+ str(counts))
    plt.xlabel('Horizontal dimension (Pixels)')
    plt.ylabel('Counts (n/pixel)')
    plt.savefig(save_name+'_XPlot.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)

    plt.show()

loc = locals()
def get_variable_name(variable):
    for k,v in loc.items():
        if loc[k] is variable:
            return k



def load_data2(RunNum):
    datainfo = data_info()
    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
        
    f = h5py.File(DataFileName, "r")
    data1 = f["/csns/instrument/module32/histogram_data"][()] #got the left bank data
    data2 = f["/csns/instrument/module31/histogram_data"][()] #got the right bank data
    #DataReshaped = np.reshape(data,(64,250,5000)i)
    ProtonCharge = f["/csns/proton_charge"][()]
    tmp = datainfo.WaveBins
    if int(RunNum[-4:]) >= 4102:
        TofPoints = 500
    else:
        TofPoints = 5000
    tmp2 = int(TofPoints/tmp)    
    tof3 = f["/csns/instrument/module31/time_of_flight"][()]
#    print(len(tof3))
    tofReshaped3 = np.average(np.reshape(tof3[:-1],(datainfo.TofBins,tmp2)),axis = 1)/1000 
    Data1Reshaped = np.reshape(data1,(64,250,tmp,tmp2))
    Data1Reshaped2 = np.sum(Data1Reshaped,axis = 3)
    Data2Reshaped = np.reshape(data2,(64,250,tmp,tmp2))
    Data2Reshaped2 = np.sum(Data2Reshaped,axis = 3) 
#    Data1Reshaped = np.reshape(data1,(64,250,5000))
#    Data2Reshaped = np.reshape(data2,(64,250,5000))
    DataStacked = np.vstack((Data2Reshaped2,Data1Reshaped2))
    #print(DataStacked[DataStacked<=0])
#    DataStacked[:,:,90:109] =0
#    DataStacked[:,:,0:10] =0       
#    print('The run number is : ' + str(RunNum))
#    print('The proton charge is: ' + str(ProtonCharge) + '\n')
    NormedCounts1, NormedCounts2 = np.sum(Data1Reshaped)/ProtonCharge*9000*390701, np.sum(Data2Reshaped)/ProtonCharge*9000*390701
    print('Total Normed counts of \nD31: ' + str(NormedCounts1) + '\nD32: ' + str(NormedCounts2) )

    Counts1, Counts2 = np.sum(Data1Reshaped)*9000, np.sum(Data2Reshaped)*9000
    print('Total counts of \nD31: ' + str(Counts1) + '\nD32: ' + str(Counts2) )
    #DataStacked[:,:,15:] = 0
    return DataStacked*9000*390701/ProtonCharge,tofReshaped3


DataFold = r'/data/hanzehua/vsanstrans'
RunInfoFold = r'/data/zuotaisen/VSANS_data_reduce/RunInfo'

PlusPlus = sys.argv[1]
PlusMinus = sys.argv[2]
#MinusMinus = sys.argv[3]
#MinusPlus = sys.argv[4]

#RunNumBkg = sys.argv[2]
#RunNum = r'RUN0002427'  #r'RUN0001827'
#print(RunNumBkg)
Info = data_info()
PP,tofPP = load_data2(PlusPlus)
PM,tofPM = load_data2(PlusMinus)
#MM,tofMM = load_data2(MinusMinus)
#MP,tofMP = load_data2(MinusPlus)
#MM,tofMM = load_data2(MinusMinus)
#plt.plot(np.sum(np.sum(PP,axis = 0),axis = 0))
#plt.show()
PP = np.sum(np.sum(PP,axis = 0),axis = 0) 
PM = np.sum(np.sum(PM,axis = 0),axis = 0)
#MM = np.sum(np.sum(MM,axis = 0),axis = 0)
#MP = np.sum(np.sum(MP,axis = 0),axis = 0)

#tmp1 = (PP - PM)/(PP + PM)
#tmp2 = (MM - MP)/(MM + MP)
#print(PP[PP>0])
output = PM/PP #np.sum(np.sum(tmp1/tmp2,axis = 0),axis = 0)
#for i in range(len(output)):
#    print(output[i])
info = data_info()
datainfo = data_info()
WavelengthArray = datainfo.const*tofPP/datainfo.D3ToMod
#WavePlot = np.sum(np.sum(DataStacked,axis = 0),axis = 0)
#plt.plot(datainfo.WavelengthArray,output)
#plt.show()
def linear_func(k,b,x):
    return k*x+b

StartFit = 10
StopFit = int(len(output) - 10)
X = WavelengthArray[StartFit:StopFit]
Y = output[StartFit:StopFit]

paras,err=scipy.optimize.curve_fit(lambda k,b,X: linear_func(k,b,X),X,Y,method = 'dogbox')

DataX = WavelengthArray #np.arange(len(output))
outputFit = paras[1]*DataX + paras[0]

with open('Trans' + PlusMinus + '.dat','w') as f:
    for i in range(len(output)):
        print(WavelengthArray[i],'  ',output[i], '    ', outputFit[i],file = f)

plt.plot(datainfo.WavelengthArray,output)
plt.plot(datainfo.WavelengthArray,outputFit)
plt.savefig('Trans' + PlusMinus + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
#plt.show()

