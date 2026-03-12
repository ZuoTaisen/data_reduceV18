import sys
import os
import re
import numpy as np

#DataFile = os.path.abspath('../') + '/' + 'test.txt'
class instrument_info():
    def __init__(self,DataFile):
        self.DataFile = DataFile
        self.L1 = float(self.get_data_dict()['L1']) 
        self.SamplePos = (float(self.get_data_dict()['SamplePos']))  #+ 80 
        self.SampleToDetBlock = 112.5 #987 #112.5   # 样品到散射腔挡板之间的距离mm，自动换样器的情况下为112.5 mm
        self.SampleDisplace = 197.5 - self.SampleToDetBlock  #85  
        self.A1 = float(self.get_data_dict()['A1'])
        self.A2 = float(self.get_data_dict()['A2'])
        self.A2Small = float(self.get_data_dict()['A2Small'])
        self.WaveMin = float(self.get_data_dict()['WaveMin'])
        self.WaveMax = float(self.get_data_dict()['WaveMax'])
        self.L1Direct = float(self.get_data_dict()['L1Direct'])
        self.A1Direct = float(self.get_data_dict()['A1Direct'])
        self.DataFold = eval(self.get_data_dict()['DataFold'])
        self.BeamStopDia = self.get_beamstop_start()
        self.IDirectBeamScale = self.L1Direct**2/self.L1**2*self.A1**2/self.A1Direct**2  #/0.785/0.83
        self.L2_factor = 1 #0.94 #0.94
        self.D1_L2 = 1000 #+ 130    #mm
        self.D2_L2 = 4000 #- 150    #mm
        self.D3_L2 = 11500    #mm
        self.D4_L2 = 12820
        self.D1Pos = self.SamplePos + self.D1_L2  #23000                #mm  D1 from moderator
        self.D2Pos = self.SamplePos + self.D2_L2  #26000               #mm  D2 from moderator
        self.D3Pos = self.SamplePos + self.D3_L2  #33500               #mm  D3 from moderator
        self.D4Pos = self.SamplePos + self.D4_L2  #34820               #mm  D4 from moderator
        self.DetFactor = 1.0
        self.TimeFixFactor = 0 #-1.70
        self.QMin = 0.001
        self.QMax = 2.5 #3.50
        self.QMax2D = 0.15#16
        self.QBins = 150 #600 #350 #120 #600#120
        self.delta0 = 0.022*(120/self.QBins)
        self.deltaL = 0.2*(120/self.QBins)
#        self.QX = self.q_generate(self.delta0,self.deltaL,self.QMin,self.QMax,self.QBins)
        self.QX = np.logspace(np.log10(self.QMin),np.log10(self.QMax),self.QBins)
#        self.QX = np.linspace(self.QMin,self.QMax,self.QBins)
        #self.TOF = 40    #ms
        self.C = 3965.2
        self.f = self.get_repetition_rate() #25 #12.5    #Hz
        self.TOF = 1000/self.f  #ms
        self.WaveBins = 250
        self.TimeDelayD1 = self.get_time_delay()[0]    #ms
        self.TimeDelayD2 = self.get_time_delay()[1]    #ms
        self.TimeDelayD3 = self.get_time_delay()[2]    #ms
        self.TimeDelayD4 = self.get_time_delay()[3]    #ms
        self.OutPath = 'output'
        self.StartWave = 0        #N^th bin  total 250
        self.StopWave = 250        # N^th  bin    total 250
        self.LambdaDivide = False
        #self.StartPoints = [2.45,4.23,4.6] #[self.WaveMin,self.WaveMin+1,self.WaveMin+2,self.WaveMin+3,self.WaveMin+4]
        #self.StopPoints = [4,4.35,self.WaveMax] #[self.WaveMin+1,self.WaveMin+2,self.WaveMin+3,self.WaveMin+4,self.WaveMax]
        self.StartPoints = [self.WaveMin]
        self.StopPoints = [self.WaveMax]
        self.WaveBinsSelectedD1 = self.get_all_wave_bins(self.StartPoints,self.StopPoints,self.TimeDelayD1,self.D1Pos)
        self.WaveBinsSelectedD2 = self.get_all_wave_bins(self.StartPoints,self.StopPoints,self.TimeDelayD2,self.D2Pos)
        self.WaveBinsSelectedD3 = self.get_all_wave_bins(self.StartPoints,self.StopPoints,self.TimeDelayD3,self.D3Pos)

        self.D2RCut = 860  # from 300 to 860 Cut the neutrons hit D2 with radius higher than D2RCut
        self.D1RCut = 860 #860 #860  # from 300 to 860 Cut the neutrons hit D1 with radius higher than D1RCut
        self.XCenter = 66 #65.20       #Pixel in D3 detector the program will calculate the XCenter and YCenter
        self.YCenter = 124       # Pixel
        self.DirectFactorSample = 1  #0.0505*1.5
        self.DirectFactorCell = 1 #0.05051*1.5
        self.point1_2_2A = 0.1 #0.04  #A
        self.point2_2_2A = 0.4    #A 
        self.point1_6A = 0.045   #0.055     #A
        self.point2_6A = 0.18     #A
        self.point1_1A = 0.05 #0.08 #0.1
        self.point2_1A = 0.018 #0.25 #0.4
        self.StitchedPoint1Auto = False #True #True
        self.StitchedPoint2Auto = False #True # True
        self.StitchPoint1 = self.get_point1()
        self.StitchPoint2 = self.get_point2()
        self.StitchAndThenSubCell = False
        self.adjustD2 = True
        self.adjustD1 = True
        self.IncludeD1 = True
        self.CellScatteringFactor = 1 #1.6978 #0.589 #1 # 1.0
        self.SampleScatteringFactor = 1 #1 #0.8 #1.6978
        self.ExtraBkg = 0
        self.DataReduce2D = False

        ########GISANS parameters
        self.GISANS_mode = False#False#False
        self.GIQYMin = 0.002
        self.GIQYMax = 0.01
        self.GIQXMin = 0.002
        self.GIQXMax = 0.01
        ########
        self.DataReduce2DScaleLog = True #False
        self.debug = False #True
        self.debug2 = False 
        self.MaskSwitch = False#False
        self.igor_format = False #True
        self.DesmearData = False #True
 
    def get_repetition_rate(self):
        WaveBand = self.WaveMax - self.WaveMin
        Frequency = 25   #Hz
        if WaveBand > 9:
            return Frequency/3
        if WaveBand > 4.5 and WaveBand <= 9:
            return Frequency/2
        if WaveBand >0 and WaveBand <= 4.5:
            return Frequency

    def get_data_dict(self):
        with open(self.DataFile, 'r') as f:
            line = []
            for sline in f.readlines():
                line.append(sline.split())
        output = {}
        for item in line:
            try:
                if item[1] == '=':
                    output[item[0]] = item[2]
            except IndexError:
                continue
        return output

    def get_beamstop_start(self):
        BeamStopDia = 50 + 17
        if self.L1 == 2490:
            self.A1 = 25
            BeamStopDia = 230 +31                  #65   #PHI 230 beamstop 
        elif self.L1 == 5150:
            BeamStopDia = 140 + 24 #42              #PHI 140 beamstop default 39  40 42
        elif self.L1 == 6730 and self.A2 <= 8:
            BeamStopDia = 90 + 22                   #25   #PHI 90 beamstop  14
        elif self.L1 == 8310 and self.A2 > 8:
            BeamStopDia = 90 + 22                   #25    #PHI 90 beamstop
        elif self.L1 == 8310 and self.A2 <= 8:
            BeamStopDia = 70 + 10 #20               #PHI 70 beamstop
        elif self.L1 == 9920 and self.A2 > 8:
            BeamStopDia = 90 + 22 #25               #PHI 90 beamstop
        elif self.L1 == 9920 and self.A2 <= 8:
            BeamStopDia = 70 + 10 #20               #PHI 70 beamstop
        elif self.L1 == 12750 and self.A2 > 8:
            BeamStopDia = 70 + 10 #18 #18          #PHI 70 beamstop 18
        elif self.L1 == 12750 and self.A2 <= 8:
            BeamStopDia = 50 + 17 #16 #18          #16  15   #PHI 50 beamstop  14
        return BeamStopDia

    def get_time_delay(self):
        #C = 3956.2
        #f = 25                 #Hz Pulse repetition rate of CSNS
        if self.WaveMin < 4:
#            D1Factor = 0.150
#            D2Factor = 0.3
#            D3Factor = 0.1
            D1Factor = 0.0
            D2Factor = 0.0
            D3Factor = 0.0
        else:

            D1Factor = 0 #-0.25
            D2Factor = 0 #-0.25 #0.2
            D3Factor = 0 #-0.6 #-0.2
        TimeDelayD1 = self.D1Pos*(self.WaveMin + self.WaveMax)/2/self.C - 1000/2/self.f     #ms
        TimeDelayD2 = self.D2Pos*(self.WaveMin + self.WaveMax)/2/self.C - 1000/2/self.f     #ms
        TimeDelayD3 = self.D3Pos*(self.WaveMin + self.WaveMax)/2/self.C - 1000/2/self.f    #ms
        TimeDelayD4 = self.D4Pos*(self.WaveMin + self.WaveMax)/2/self.C - 1000/2/self.f    #ms
        FixFactorD3 = self.TimeFixFactor + D3Factor  # 1.8 #0.06*17.4 #TimeDelayD3
        FixFactorD1 = self.TimeFixFactor + D1Factor #0.05 #*(self.SamplePos+self.D1_L2)/(self.SamplePos+self.D3_L2)
        FixFactorD2 = self.TimeFixFactor + D2Factor # 0.1   #*(self.SamplePos+self.D2_L2)/(self.SamplePos+self.D3_L2)
        if TimeDelayD1 <= 0:
            TimeDelayD1 = 0.1
        if TimeDelayD2 <= 0:
            TimeDelayD2 = 0.1
        if TimeDelayD3 <= 0:
            TimeDelayD3 = 0.1
        return TimeDelayD1+FixFactorD1,TimeDelayD2+FixFactorD2,TimeDelayD3 + FixFactorD3,TimeDelayD4 + self.TimeFixFactor

    def q_generate(self,delta0,deltaL,Qmin,Qmax,points):
        Q0 = delta0*np.tanh(deltaL*Qmin/delta0)
        Q = np.zeros(points)    
        Q[0] = Qmin
        for i in range(1,len(Q)):
            if Q[i] <= Qmax:
                Qi = delta0*np.tanh(deltaL*Q[i-1]/delta0)
                Q[i] = Qi + Q[i-1]  
        return Q

    def get_wave_bins(self,StartPoint,StopPoint,TimeDelayDN,DetPos):
        Bins = np.arange(self.StartWave,self.StopWave)
        TofArrayDN = np.linspace(TimeDelayDN,TimeDelayDN + self.TOF,self.WaveBins)
        WavelengthArrayDN = self.C*TofArrayDN/DetPos
        return Bins[(WavelengthArrayDN >= StartPoint)*(WavelengthArrayDN < StopPoint)]

    def get_all_wave_bins(self,StartPoints,StopPoints,TimeDelay,DetPos):
        Bins = []
        for i in range(len(StartPoints)):
            tmp = self.get_wave_bins(StartPoints[i],StopPoints[i],TimeDelay,DetPos)
            Bins.append(tmp)
        BinsOut = Bins #np.array(Bins) #np.concatenate(Bins)
        #print(BinsOut)
        return BinsOut

    def get_point1(self):
        point1_2_2A = self.point1_2_2A
        point1_6A = self.point1_6A 
        point1_1A = self.point1_1A
        if self.StitchedPoint1Auto is True:
            point1,_ = self.get_stitch_point_auto()
        elif self.WaveMin == 2.2:
            point1 = point1_2_2A
        elif self.WaveMin == 6:
            point1 = point1_6A
        elif self.WaveMin == 4.5:
            point1 = 0.045 #0.07
        elif self.WaveMin == 1:
            point1 = point1_1A #0.12
        else:
            tmp = 1/2.2-1/6
            k = (point1_2_2A-point1_6A)/tmp
            b = k/2.2*(-1) + point1_2_2A
            point1 = k/self.WaveMin + b
        return point1

    def get_point2(self):
        point2_2_2A = self.point2_2_2A
        point2_6A = self.point2_6A
        point2_1A = self.point2_1A
        if self.StitchedPoint2Auto is True:
            _,point2 = self.get_stitch_point_auto()
        elif self.WaveMin == 2.2:
            point2 = point2_2_2A
        elif self.WaveMin == 6:
            point2 = point2_6A
        elif self.WaveMin == 4.5:
            point2 = 0.16 #0.4
        elif self.WaveMin == 1:
            point2 = point2_1A #0.4
        else:
            tmp = 1/2.2-1/6
            k = (point2_2_2A-point2_6A)/tmp
            b = k/2.2*(-1) + point2_2_2A
            point2 = k/self.WaveMin +b
        return point2
            
    def q_calc(self,Theta,Wave):
        return 4*np.pi*np.sin(Theta/2)/Wave

    def get_mod(self,input_file):
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for line in lines:
                if 'python' in line and not line.strip().startswith('#'):
                    match = re.search(r"'([^']*)'", line)
                    if match:
                        content = match.group(1)
                        if '-V' in content and 'deg' in content:
                            return 'vertical'
                        elif '-H' in content and 'deg' in content:
                            return 'horizontal'
                        else:
                            return 'average'
        infile.close()

    def get_stitch_point_auto(self):
        HoleDia = 300
        D1D2Dia = 700
        D3Dia = 500
        ThetaD3 = np.arctan(D3Dia/self.D3_L2)
        ThetaD3Diagonal = np.arctan(D3Dia*np.sqrt(2)/self.D3_L2)
        ThetaD2InTop = np.arctan(HoleDia/(self.D2_L2+430))
        ThetaD2OutTop = np.arctan(D1D2Dia/(self.D2_L2+430)) 
        ThetaD1InTop = np.arctan(HoleDia/(self.D1_L2+430))
        #ThetaD1OutTop = np.arctan(D1D2Dia/(self.D1_L2+430))       
        ThetaD2InSide = np.arctan(HoleDia/(self.D2_L2))
        ThetaD2OutSide = np.arctan(D1D2Dia/(self.D2_L2))
        ThetaD1InSide = np.arctan(HoleDia/(self.D1_L2))
        #ThetaD1OutSide = np.arctan(D1D2Dia/(self.D1_L2))
        QD3 = self.q_calc(ThetaD3,self.WaveMin)
        QD3Diagonal = self.q_calc(ThetaD3Diagonal,self.WaveMin)
        QD2InTop = self.q_calc(ThetaD2InTop,self.WaveMax)
        QD2InSide = self.q_calc(ThetaD2InSide,self.WaveMax)
        QD2OutTop = self.q_calc(ThetaD2OutTop,self.WaveMin)
        QD2OutSide = self.q_calc(ThetaD2OutSide,self.WaveMin)
        QD1InTop = self.q_calc(ThetaD1InTop,self.WaveMax)
        QD1InSide = self.q_calc(ThetaD1InSide,self.WaveMax)       
        StitchPoint1Top = (QD3 + QD2InTop)/2
        StitchPoint1Side = (QD3 + QD2InSide)/2
        StitchPoint2Top = (QD2OutTop + QD1InTop)/2
        StitchPoint2Side = (QD2OutSide + QD1InSide)/2
        #if int(sys.version.split('|')[0].split('.')[1])>=12:
        #    mod = self.get_mod('./batchrun.py')
        #else:
        mod = self.get_mod('./batchrun.py')
        #print(StitchPoint1Top,StitchPoint1Side,StitchPoint2Top,StitchPoint2Side)
        if mod == 'vertical':
            return  StitchPoint1Top, StitchPoint2Top
        elif mod == 'horizontal':
            return StitchPoint1Side, StitchPoint2Side
        else:
            #print((StitchPoint1Top+StitchPoint1Side)/2, (StitchPoint2Top+StitchPoint2Side)/2)
            return (StitchPoint1Top+StitchPoint1Side)/2, (StitchPoint2Top+StitchPoint2Side)/2


