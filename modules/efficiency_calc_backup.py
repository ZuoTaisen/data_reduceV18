import numpy as np
import matplotlib.pyplot as plt
import intersection_deepseek_optimized as section
global test, test2
class efficiency():
    def __init__(self,DetDistance,XPos,YPos,wavelength):
        self.XPos = XPos
        self.YPos = YPos
        self.DetDistance = DetDistance
        self.Phi = np.arctan(XPos/DetDistance)
        self.Zeta = np.arctan(YPos/DetDistance)
        self.TubeRadius = 3.6 # mm
        self.TubePixelSize = 4  # mm
        self.TubeDistance = 8.5  # mm
        self.RadiusPoints = 21
        self.PixelPoints = 31#171
        self.diagonal = np.sqrt((self.TubeRadius*2)**2 + self.TubePixelSize**2)
        self.rArray = np.linspace(-1*self.TubeRadius, self.TubeRadius,self.RadiusPoints)
        self.SampleToTube = np.sqrt(self.DetDistance**2 + XPos**2)
        self.RMin = self.SampleToTube[:,None] - np.sqrt((self.TubeRadius)**2 - self.rArray**2)
        self.RMax = self.SampleToTube[:,None] + np.sqrt((self.TubeRadius)**2 - self.rArray**2)        
        
        self.wavelength = wavelength
        self.pressure = 20 # atm
        self.AlAbsorb = 0.97

        # self.d_pixel = np.sqrt((self.DetDistance*np.tan(self.Phi)) + (self.DetDistance/np.cos(self.Zeta)))
        # self.d_XZ = self.DetDistance*np.tan(self.Phi)

    def zeta_i(self,i):
        ZetaI = np.arctan(self.YPos[i]/self.SampleToTube) #self.DetDistance)
        return ZetaI
    
    def zeta_array(self,i):
        tmp = np.linspace(-1*self.diagonal/2/self.DetDistance,self.diagonal/2/self.DetDistance,self.PixelPoints)
        ZetaArray = self.zeta_i(i)[:,None] + np.arctan(tmp)
        return ZetaArray
        
    def phi_array(self):
        PhiArray = self.Phi[:,None] + np.arctan(self.rArray/self.SampleToTube[:,None])
        return PhiArray

    def rects(self,i,ymin,ymax):
        xmin = self.RMin
        xmax = self.RMax
        return xmin, xmax, ymin, ymax

    def intersection(self,i):
        ymin = (self.YPos[i] - self.TubePixelSize/2)*np.ones_like(self.RMin)
        ymax = (self.YPos[i] + self.TubePixelSize/2)*np.ones_like(self.RMax) 
        xmin, xmax, ymin, ymax = self.rects(i,ymin,ymax)
        zeta_array = self.zeta_array(i)
        # 调整输入形状以匹配重构后的函数接口
        # zeta_array: (m, n)
        # xmin, xmax, ymin, ymax: (m, k)
        k_array = np.tan(zeta_array)
        LengthTensor = section.batch_line_segments_optimized(k_array, xmin, xmax, ymin, ymax)
        return LengthTensor

    def intersection_long(self,i):
        if np.sign(self.YPos[0]) > 0:
            ymin = (self.YPos[i] - 4*self.TubePixelSize/2)*np.ones_like(self.RMin)
            ymax = (self.YPos[i] + self.TubePixelSize/2)*np.ones_like(self.RMax) 
        if np.sign(self.YPos[0]) <= 0:
            ymin = (self.YPos[i] - self.TubePixelSize/2)*np.ones_like(self.RMin)
            ymax = (self.YPos[i] + 4*self.TubePixelSize/2)*np.ones_like(self.RMax) 
        xmin, xmax, ymin, ymax = self.rects(i,ymin,ymax)
        zeta_array = self.zeta_array(i)
        # 调整输入形状以匹配重构后的函数接口
        # zeta_array: (m, n)
        # xmin, xmax, ymin, ymax: (m, k)
        k_array = np.tan(zeta_array)
        LengthTensor = section.batch_line_segments_optimized(k_array, xmin, xmax, ymin, ymax)
        return LengthTensor

    def intersection_nei(self,i):
        if np.sign(self.YPos[0]) > 0:
            ymin = (self.YPos[i] - 4*self.TubePixelSize/2)*np.ones_like(self.RMin)
            ymax = (self.YPos[i] - self.TubePixelSize/2)*np.ones_like(self.RMax) 
        if np.sign(self.YPos[0]) <= 0:
            ymin = (self.YPos[i] + self.TubePixelSize/2)*np.ones_like(self.RMin)
            ymax = (self.YPos[i] + 4*self.TubePixelSize/2)*np.ones_like(self.RMax) 
        xmin, xmax, ymin, ymax = self.rects(i,ymin,ymax)
        zeta_array = self.zeta_array(i)
        # 调整输入形状以匹配重构后的函数接口
        # zeta_array: (m, n)
        # xmin, xmax, ymin, ymax: (m, k)
        k_array = np.tan(zeta_array)
        LengthTensor = section.batch_line_segments_optimized(k_array, xmin, xmax, ymin, ymax)
        return LengthTensor
    
    def intersection_nei_x(self,t):
        PhiArray = self.phi_array()
        length = self.intersection(t)  
        length2 = self.intersection_long(t) 
        for i in range(len(PhiArray)):            
            tmp = PhiArray[i]
            if tmp[10] < 0 and i < len(PhiArray)-1: 
                index = len(tmp[tmp >= PhiArray[i+1][0]])
                if index == 0:
                    length[i,:,:] = 0
                elif index > 0:
                    #print(index)
                    #length[i,:,:] = 0
                    length[i,:,:index*(-1)] = 0
                    length[i,:,index*(-1):] = length2[i+1,:,:index]
            elif i == len(PhiArray)-1:
                length[i,:,:] = 0
        for i in range(len(PhiArray)):
            tmp = PhiArray[i]
            if tmp[10] > 0 and i >= 1: 
                index = len(tmp[tmp <= PhiArray[i-1][-1]])
                if index == 0:
                    length[i,:,:] = 0
                elif index > 0:
                    #print(index)
                    length[i,:,index:] = 0
                    length[i,:,:index] = length2[i-1,:,index*(-1):]
            elif i == 0:
                length[i,:,:] = 0
        return length                  
                    
    
    def he3_absorb(self,thickness):
        # 预计算系数以减少重复计算
        coeff = -0.0732 * self.pressure
        # 恢复维度扩展以确保正确的广播
        return 1 - np.exp(coeff * self.wavelength * thickness[:, :, :, None])

    def he3_trans(self,thickness):
        # 预计算系数以减少重复计算
        coeff = -0.0732 * self.pressure
        # 恢复维度扩展以确保正确的广播
        return np.exp(coeff * self.wavelength * thickness[:, :, :, None])

    def h_pixel_absorb(self,i):
        GasThickness = self.intersection(i)
        absorb = self.he3_absorb(GasThickness)#*self.AlAbsorb**3
        return absorb

    def h_pixel_trans(self,i):
        GasThicknessV = self.intersection_nei(i)
        GasThicknessH = self.intersection_nei_x(i)        
        TransV = self.he3_trans(GasThicknessV)#*self.AlAbsorb**3
        TransH = self.he3_trans(GasThicknessH)#*self.AlAbsorb**3
        trans = TransV*TransH#*
        return trans

    def h_pixel_efficiency(self,i):
        absorb = self.h_pixel_absorb(i)
        trans = self.h_pixel_trans(i)
        tmp = absorb*trans
        #points = np.product(tmp.shape)
        efficiency = np.average(np.average(tmp,axis = 2),axis = 1)        
        return efficiency

    def pixel_efficiency(self):
        eff = []
        for i in range(len(self.YPos)):
            eff_i = self.h_pixel_efficiency(i)
            eff.append(eff_i)
        out = np.transpose(np.array(eff),(1,0,2))
        return out

    def save_matrix(self,matrix,FileName):
        with open(FileName,'w') as f:
            for i in matrix:
                for j in i:
                    print('{:<20.8f}'.format(j),file = f, end = '')
                print('\n',file = f, end = '')
        f.close()

    def plot_data_2d(self,QX,QY,QArray,show = True,FileName = '2D_plot', logscale = False):
        if logscale == True:
            QArray[QArray == np.nan] = 1E-10
            #QArray[QArray < 0] = 1E-10
            QArray = np.log(np.abs(QArray))
        plt.figure(figsize=(5, 5))
        plt.contour(QX,QY,QArray,600,cmap = 'hot')
        plt.xlabel('QX (Angstrom$^{-1}$)')
        plt.ylabel('QY (Angstrom$^{-1}$)')
        plt.savefig(FileName + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
    
    def mat_plot_2d(self,QArray,show = True,FileName = 'MatPlot2D', logscale = False):
        if logscale == True:
            #QArray[QArray == np.nan] = 1E-10
            #QArray[QArray <= 0] = 1E-10
            QArray = QArray + 1E-8
            QArray = np.log(QArray)
        plt.figure(figsize=(5, 5))
        plt.matshow(QArray,cmap = plt.cm.jet)
        plt.xlabel('QX')
        plt.ylabel('QY')
        plt.savefig(FileName + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()


if __name__ == "__main__":
    DetDistance = 1000 #mm
    TubeRadius = 4 #mm
    
    wavelength = np.linspace(2,11,3)
    XPos = np.linspace(-710,710,176)
    YPos = np.linspace(-500,500,250)
    PosMatrix = XPos*YPos[:,None]

    
    
    
    calc = efficiency(DetDistance,XPos,YPos,wavelength)
    #pixel_eff = calc.h_pixel_efficiency(0)
    pixel_eff = calc.pixel_efficiency()
    calc.mat_plot_2d(np.sum(pixel_eff,axis = 2))
    
    
    #length = calc.intersection(0)
    
    
    
    # matrix = np.average(pixel_efficiency,axis = 2)
    # calc.mat_plot_2d(matrix)










