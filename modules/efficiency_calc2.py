import numpy as np
import matplotlib.pyplot as plt

class efficiency():
    def __init__(self,DetDistance,XPos,YPos,wavelength):
        #self.YPos = YPos
        #self.DetDistance = DetDistance
        self.DetDistance = DetDistance
        self.Phi0 = np.arctan(XPos/DetDistance)
        self.Zeta0 = np.arctan(YPos/DetDistance)

       
        self.Phi = self.Phi0[:,None][:,None][:,None]  # Rad
        self.Zeta = self.Zeta0[:,None][:,None]  # Rad
        self.wavelength = wavelength
        self.pressure = 20 # atm
        self.AlAbsorb = 0.97
        self.factor = 140 if np.min(wavelength) < 4 else -1

        self.TubeRadius = 3.6 # mm
        self.TubePixelSize = 4  # mm
        self.TubeDistance = 8.5  # mm
        self.DeltaPhiPlus = np.arctan((self.DetDistance * np.tan(self.Phi) - self.TubeDistance) / self.DetDistance) - self.Phi # psai[0]
        self.psai0 = self.TubeRadius/self.DetDistance * 1.0
        self.psai = np.linspace(self.psai0 * (-1), self.psai0, 20)[:, None]
        self.d_pixel = np.sqrt((self.DetDistance*np.tan(self.Phi)) + (self.DetDistance/np.cos(self.Zeta)))
        self.d_XZ = self.DetDistance*np.tan(self.Phi)


    def gas_thickness(self,psai):
        ThickSquare = self.TubeRadius ** 2 - (self.DetDistance * np.sin(psai)) ** 2
        ThickSquare[ThickSquare<0] = 0
        thickness = 2*np.sqrt(ThickSquare)*(np.cos(self.Zeta))**self.factor
        #print(np.cos(self.Zeta)**10)
        #print(self.YPos)
        #print(self.DetDistance)
        return thickness/10

    def he3_absorb(self,thickness):
        absorb = 1-np.exp(-0.0732*self.pressure*self.wavelength*thickness)
        return absorb

    def he3_trans(self,thickness):
        trans = np.exp(-0.0732*self.pressure*self.wavelength*thickness)
        return trans

    def single_tube_absorb(self,psai):
        GasThickness = self.gas_thickness(psai)
        absorb = self.he3_absorb(GasThickness)*self.AlAbsorb
        #print(absorb)
        return absorb

    def single_tube_trans(self,psai):
        GasThickness = self.gas_thickness(psai)
        trans = self.he3_trans(GasThickness)*self.AlAbsorb**3
        return trans

    def pixel_efficiency(self):
        absorb = self.single_tube_absorb(self.psai)
        trans = self.single_tube_trans(self.psai + self.DeltaPhiPlus)
        #return np.trapz(absorb*trans,np.ravel(self.psai),axis = 2)
        return np.average(absorb*trans,axis = 2)

    def i_pm(self,plus):
        if plus is True:
            IPm = (self.d_pixel*np.sin(self.psai) + self.TubePixelSize)/np.tan(self.psai) - self.d_XZ
        else:
            IPm = (self.d_pixel * np.sin(self.psai) - self.TubePixelSize) / np.tan(self.psai) - self.d_XZ
        return IPm

    def s_in_out(self,plus = True):
        SPhiPsai = self.gas_thickness(self.psai)
        IPm = self.i_pm(plus)
        DeltaPm = self.TubeRadius + IPm
        TPm = (self.TubeRadius - DeltaPm)/np.cos(self.Phi)
        OPhi = self.d_pixel/np.cos(self.Phi) - self.d_pixel*np.cos(self.Phi)
        LPm = TPm*(TPm<=OPhi) - OPhi*(OPhi<TPm)
        S1 = SPhiPsai/2 - LPm
        S2 = SPhiPsai/2 + LPm
        return S1,S2


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

# DetDistance = 1000 #mm
# TubeRadius = 4 #mm

# wavelength = np.linspace(2,11,20)
# XPos = np.linspace(-700,700,128)
# YPos = np.linspace(-700,700,250)
# PosMatrix = XPos*YPos[:,None]
# Phi = np.arctan(XPos/DetDistance)
# Zeta = np.arctan(YPos/DetDistance)



# calc = efficiency(DetDistance,Phi,Zeta,wavelength)
# pixel_efficiency = calc.pixel_efficiency()



# matrix = np.average(pixel_efficiency,axis = 2)
# calc.mat_plot_2d(matrix)










