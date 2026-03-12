import sys
import os
import numpy as np
cpath = os.getcwd()
sys.path.append(cpath + r'/modules')
import sample_reader
import importlib as imp
import D3
import D2
import D1
import instrument_reader
#import multiprocessing
instrument_info = instrument_reader.instrument_info(sys.argv[1])

try:
    sys.argv[3]
except NameError:
    var_exists = False
else:
    var_exists = True

if var_exists == True:
    samples = sample_reader.read(sys.argv[2])[int(sys.argv[3]):int(sys.argv[4])]
else:
    samples = sample_reader.read(sys.argv[2])[:]  # Choose the samples with [1:2] or [1:5] [0:3]

for i in range(len(samples)):
    print(samples[i])

LambdaDivided = False 
tmp1 = instrument_info.WaveBinsSelectedD1
tmp2 = instrument_info.WaveBinsSelectedD2
tmp3 = instrument_info.WaveBinsSelectedD3
OutPath = sys.argv[5]
if instrument_info.DataReduce2D is True:
    for sample in samples:
        if not os.path.exists(OutPath):
            os.makedirs(OutPath)
            os.makedirs(OutPath + '/' + 'Transmission')
        instrument_info.OutPath = OutPath
        instrument_info.WaveBinsSelectedD1 = np.concatenate(tmp1)
        instrument_info.WaveBinsSelectedD2 = np.concatenate(tmp2)
        instrument_info.WaveBinsSelectedD3 = np.concatenate(tmp3)
    
        imp.reload(D3)
    
        XCenter,YCenter = D3.get_center(samples[0][5],instrument_info)
        instrument_info.XCenter = XCenter
        instrument_info.YCenter = YCenter
        imp.reload(D3)
        D3.reduce_2d(sample,instrument_info)

else:    
    for sample in samples:
        if LambdaDivided is False:
             
            if not os.path.exists(OutPath):
                os.makedirs(OutPath)
                os.makedirs(OutPath + '/' + 'Transmission')
            if not os.path.exists(OutPath + '/' + 'StitchedDataOnlySample'):
                os.makedirs(OutPath + '/' + 'StitchedDataOnlySample')
            if not os.path.exists(OutPath + '/' + 'StitchedDataOnlyCell'):
                os.makedirs(OutPath + '/' + 'StitchedDataOnlyCell')
            if not os.path.exists(OutPath + '/' + 'StitchedDataSampleCell'):
                os.makedirs(OutPath + '/' + 'StitchedDataSampleCell')
            instrument_info.OutPath = OutPath
            instrument_info.WaveBinsSelectedD1 = np.concatenate(tmp1)
            instrument_info.WaveBinsSelectedD2 = np.concatenate(tmp2)
            instrument_info.WaveBinsSelectedD3 = np.concatenate(tmp3)
        
            imp.reload(D3)
            
            XCenter,YCenter = D3.get_center(samples[0][5],instrument_info)
            instrument_info.XCenter = XCenter
            instrument_info.YCenter = YCenter
            imp.reload(D3)

            D3.reduce(sample,instrument_info)
            imp.reload(D2)
            D2.reduce(sample,instrument_info)
            imp.reload(D1)
            D1.reduce(sample,instrument_info)

        
        elif LambdaDivided is True:
            OutPath0 = OutPath
            #print(instrument_info.WaveBinsSelectedD1)
            for t in range(len(instrument_info.WaveBinsSelectedD3)):
                OutPath = OutPath0 +'_' + str(t) #str(instrument_info.StartPoints[t]) + '-' + str(instrument_info.StopPoints[t])
                if not os.path.exists(OutPath):
                    os.makedirs(OutPath)
                    os.makedirs(OutPath + '/' + 'Transmission')
                if not os.path.exists(OutPath + '/' + 'StitchedDataOnlySample'):
                    os.makedirs(OutPath + '/' + 'StitchedDataOnlySample')
                if not os.path.exists(OutPath + '/' + 'StitchedDataOnlyCell'):
                    os.makedirs(OutPath + '/' + 'StitchedDataOnlyCell')
                if not os.path.exists(OutPath + '/' + 'StitchedDataSampleCell'):
                    os.makedirs(OutPath + '/' + 'StitchedDataSampleCell')
                instrument_info.OutPath = OutPath
    
                instrument_info.WaveBinsSelectedD1 = tmp1[t]
                instrument_info.WaveBinsSelectedD2 = tmp2[t]
                instrument_info.WaveBinsSelectedD3 = tmp3[t]
        
                imp.reload(D3)
        
                XCenter,YCenter = D3.get_center(samples[0][5],instrument_info)
                instrument_info.XCenter = XCenter
                instrument_info.YCenter = YCenter
                imp.reload(D3)
                D3.reduce(sample,instrument_info)
                imp.reload(D2)
                D2.reduce(sample,instrument_info)
                imp.reload(D1)
                D1.reduce(sample,instrument_info)

