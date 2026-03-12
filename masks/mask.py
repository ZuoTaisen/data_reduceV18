import sys
import os
import numpy as np
cpath = os.getcwd()
sys.path.append(r'modules')
import sample_reader
import importlib as imp
import D3
import D2
import D1
import instrument_reader

###D3 dimension: X*Y = 128*250
#XMinD3 = [0,  0,   63,75,81,33,53,37,67,53,73,26]          # Lower bond of X pixels to be masked
##XMaxD3 = [128, 128,71,81,87,40,59,41,77,58,77,33]          # Higher bond of X pixels to be masked
#YMinD3 = [0,249,74,46,56,127,63,143,193,32,106,79]                # Lower bond of Y pixels to be masked
#YMaxD3 = [1,250,85,57,67,143,76,155,202,37,117,87]                # Higher bond of Y pixels to be mased
XMinD3 = [49,62]
XMaxD3 = [59,78]
YMinD3 = [168,130]
YMaxD3 = [191,1438]
#XMinD3 = [0,0,44.9,51.5,53.7,88]
#XMaxD3 = [128,128,65,63.5,69.7,100]
#YMinD3 = [0,249,161,83,155.4,106]
#YMaxD3 = [1,250,198,109.5,192.1,139]

#XMinD3 = [0,0,55,60,59,61,59,82,49,46,80,80,75,52,70]
#XMaxD3 = [128,128,65,66,64,66,65,84,54,55,86,86,79,64,75]
#YMinD3 = [0,249,167,95,167,98,192,128,139,155,90,156,138,182,61]
#YMaxD3 = [1,250,176,107,178,107,205,136,139,168,98,165,146,198,72]

#XMinD3 = [0,0,62,46]
##XMaxD3 = [128,128,70,53]
#YMinD3 = [0,249,103,157]
#YMaxD3 = [1,250,118,175]

#XMinD3 = [0,0,43,66,61,66,45,56,35,40,20,49,62,45]
#XMaxD3 = [128,128,53,73,70,81,52,59,43,49,28,54,67,50]
#YMinD3 = [0,249,137,141,94,139,121,123,159,168,122,200,103,85]
#YMaxD3 = [1,250,183,155,116,157,134,132,172,188,137,228,115,100]

###D21 and D23 dimension: X*Y = 250*48  #*250
###D22 and D24 dimension: X*Y = 48*150
XMinD21 = [0,247]          # Lower bond of X pixels to be masked
XMaxD21 = [3,250]          # Higher bond of X pixels to be masked
YMinD21 = [0,0]                # Lower bond of Y pixels to be masked
YMaxD21 = [48,48]                # Higher bond of Y pixels to be mased
XMinD22 = [0,0]          # Lower bond of X pixels to be masked
XMaxD22 = [48,48]          # Higher bond of X pixels to be masked
YMinD22 = [0,117]                # Lower bond of Y pixels to be masked
YMaxD22 = [3,150]                # Higher bond of Y pixels to be mased
XMinD23 = [0,247]          # Lower bond of X pixels to be masked
XMaxD23 = [3,250]          # Higher bond of X pixels to be masked
YMinD23 = [0,0]               # Lower bond of Y pixels to be masked
YMaxD23 = [48,48]              # Higher bond of Y pixels to be mased
XMinD24 = [0,0]             # Lower bond of X pixels to be masked
XMaxD24 = [48,48]           # Higher bond of X pixels to be masked
YMinD24 = [0,147]               # Lower bond of Y pixels to be masked
YMaxD24 = [3,150]               # Higher bond of Y pixels to be mased
XMinD2 = {0:XMinD21,1:XMinD22,2:XMinD23,3:XMinD24}
XMaxD2 = {0:XMaxD21,1:XMaxD22,2:XMaxD23,3:XMaxD24}
YMinD2 = {0:YMinD21,1:YMinD22,2:YMinD23,3:YMinD24}
YMaxD2 = {0:YMaxD21,1:YMaxD22,2:YMaxD23,3:YMaxD24}

###D11 and D13 dimension: X*Y = 250*48   #*250
###D12 and D14 dimension: X*Y = 48*150
#XMinD11 = [0,246,95,90,0]#,0,220]           # Lower bond of X pixels to be masked
#XMaxD11 = [4,250,115,99,250]#,30,250]           # Higher bond of X pixels to be masked
#YMinD11 = [0,0,19,23,0]#,0,0]                 # Lower bond of Y pixels to be masked
#YMaxD11 = [48,48,23,28,48]#,25,25]               # Higher bond of Y pixels to be mased
XMinD11 = [0,246]#,0,220]           # Lower bond of X pixels to be masked
XMaxD11 = [2,250]#,30,250]           # Higher bond of X pixels to be masked
YMinD11 = [0,0]#,0,0]                 # Lower bond of Y pixels to be masked
YMaxD11 = [48,48]#,25,25]               # Higher bond of Y pixels to be mased

XMinD12 = [0,0]               # Lower bond of X pixels to be masked
XMaxD12 = [48,48]             # Higher bond of X pixels to be masked
YMinD12 = [0,146]                 # Lower bond of Y pixels to be masked
YMaxD12 = [4,150]                 # Higher bond of Y pixels to be mased

#XMinD13 = [0,246,11,0]#,0,220]           # Lower bond of X pixels to be masked
#XMaxD13 = [4,250,29,250]#,30,250]           # Higher bond of X pixels to be masked
#YMinD13 = [0,0,3,0]#,25,25]                 # Lower bond of Y pixels to be masked
#YMaxD13 = [48,48,9,48]#,48,48]               # Higher bond of Y pixels to be mased

XMinD13 = [0,246]#,0,220]           # Lower bond of X pixels to be masked
XMaxD13 = [2,250]#,30,250]           # Higher bond of X pixels to be masked
YMinD13 = [0,0]#,25,25]                 # Lower bond of Y pixels to be masked
YMaxD13 = [48,48]#,48,48]               # Higher bond of Y pixels to be mased

XMinD14 = [0,0]               # Lower bond of X pixels to be masked
XMaxD14 = [48,48]             # Higher bond of X pixels to be masked
YMinD14 = [0,146]                 # Lower bond of Y pixels to be masked
YMaxD14 = [4,150]                 # Higher bond of Y pixels to be mased
XMinD1 = {0:XMinD11,1:XMinD12,2:XMinD13,3:XMinD14}
XMaxD1 = {0:XMaxD11,1:XMaxD12,2:XMaxD13,3:XMaxD14}
YMinD1 = {0:YMinD11,1:YMinD12,2:YMinD13,3:YMinD14}
YMaxD1 = {0:YMaxD11,1:YMaxD12,2:YMaxD13,3:YMaxD14}

# Fan mask
PhiMin = []
PhiMax = []
info = instrument_reader.instrument_info(sys.argv[1])
DataFold = info.DataFold

first = int(sys.argv[3])
last = int(sys.argv[4])
samples = sample_reader.read(sys.argv[2])[first:last][0] 

XCenter,YCenter = D3.get_center(samples[5],info)
info.XCenter = XCenter
info.YCenter = YCenter
imp.reload(D3)

D3_data_reduce = D3.data_reduce(DataFold,info)
D2_data_reduce = D2.data_reduce(DataFold,info)
D1_data_reduce = D1.data_reduce(DataFold,info)

SampleName, SampleScattering,CellScattering,SampleDirect,CellDirect,AirDIrect,Samplethickness = samples
#SampleScatteringRun = r"RUN000" + SampleScattering
SampleScatteringRun = r"RUN" + str('0'*(7-len(SampleScattering.split('_')[0]))) + SampleScattering

D3Data,_,_,_ = D3_data_reduce.load_data_origin(SampleScatteringRun)
GroupingMaskD3 = D3_data_reduce.grouping_mask(D3Data,XMinD3,XMaxD3,YMinD3,YMaxD3)
AzimuthalMaskD3 = D3_data_reduce.azimuthal_mask(D3Data,PhiMin,PhiMax)
D3Mask = GroupingMaskD3*AzimuthalMaskD3
np.save('masks/D3Mask.npy',D3Mask)
np.save('masks/D3Data.npy',np.sum(D3Data,axis = 2))

D2Data,_ = D2_data_reduce.load_data_origin(SampleScatteringRun)
GroupingMaskD2 = D2_data_reduce.grouping_mask(D2Data,XMinD2,XMaxD2,YMinD2,YMaxD2)
AzimuthalMaskD2 = D2_data_reduce.azimuthal_mask(D2Data,PhiMin,PhiMax)
D2Mask = {0:GroupingMaskD2[0]*AzimuthalMaskD2[0],1:GroupingMaskD2[1]*AzimuthalMaskD2[1],2:GroupingMaskD2[2]*AzimuthalMaskD2[2],3:GroupingMaskD2[3]*AzimuthalMaskD2[3]}
np.save('masks/D2Mask0.npy',D2Mask[0])
np.save('masks/D2Mask1.npy',D2Mask[1])
np.save('masks/D2Mask2.npy',D2Mask[2])
np.save('masks/D2Mask3.npy',D2Mask[3])
np.save('masks/D2Data0.npy',np.sum(D2Data[0],axis = 2))
np.save('masks/D2Data1.npy',np.sum(D2Data[1],axis = 2))
np.save('masks/D2Data2.npy',np.sum(D2Data[2],axis = 2))
np.save('masks/D2Data3.npy',np.sum(D2Data[3],axis = 2))


D1Data,_ = D1_data_reduce.load_data_origin(SampleScatteringRun)
GroupingMaskD1 = D1_data_reduce.grouping_mask(D1Data,XMinD1,XMaxD1,YMinD1,YMaxD1)
AzimuthalMaskD1 = D1_data_reduce.azimuthal_mask(D1Data,PhiMin,PhiMax)
D1Mask = {0:GroupingMaskD1[0]*AzimuthalMaskD1[0],1:GroupingMaskD1[1]*AzimuthalMaskD1[1],2:GroupingMaskD1[2]*AzimuthalMaskD1[2],3:GroupingMaskD1[3]*AzimuthalMaskD1[3]}
np.save('masks/D1Mask0.npy',D1Mask[0])
np.save('masks/D1Mask1.npy',D1Mask[1])
np.save('masks/D1Mask2.npy',D1Mask[2])
np.save('masks/D1Mask3.npy',D1Mask[3])
np.save('masks/D1Data0.npy',np.sum(D1Data[0],axis = 2))
np.save('masks/D1Data1.npy',np.sum(D1Data[1],axis = 2))
np.save('masks/D1Data2.npy',np.sum(D1Data[2],axis = 2))
np.save('masks/D1Data3.npy',np.sum(D1Data[3],axis = 2))




