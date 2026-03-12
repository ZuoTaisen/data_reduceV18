import numpy as np
from matplotlib import pyplot as plt

class data_plot():
    def __init__(self):
        pass 

    def plot_data(self,xx,yy,label = np.arange(1,100),save = False, FileName = 'SavedPlot.png', title = 'title', xlabel = 'Lambda (Angstrom)', ylabel = 'Counts (n/s/Angstrom)',show = False, logx = False, logy 
= False):        
        for i in range(len(xx)):#(len(tof)):
            plt.plot(xx[i],yy[i],label=label[i])
        plt.rcParams.update({'font.size': 15})
        plt.rc('font',family = 'Times New Roman')
        plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        plt.tick_params(axis = 'both',direction = 'in',labelsize = 15,width = 2)
        plt.tick_params(axis = 'both',which = 'minor',direction = 'in',labelsize = 15,width = 1.5,length = 4)
        plt.legend(fancybox=True, framealpha=0.01,fontsize = 16,frameon = False,loc = 'upper right')
        plt.xlabel(xlabel,size = 20)
        plt.ylabel(ylabel,size =20)
        plt.ylim = (-50,50)
        plt.title(title)
        if logx is True:
            plt.xscale('log')
        if logy is True:
            plt.yscale('log')
        if save is True:
            plt.savefig(FileName,dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
        if show == True:
            plt.show()
        plt.close()

    def plot_data_with_errorbar(self,xx,yy,error,label = np.arange(1,100),save = False, FileName = 'SavedPlot.png', title = 'title', xlabel = 'Lambda (Angstrom)', ylabel = 'Counts (n/s/Angstrom)', logx = False
, logy = False,show = False):        
        for i in range(len(xx)):#(len(tof)):
            plt.errorbar(xx[i],yy[i],error[i],label=label[i])
        plt.rcParams.update({'font.size': 15})
        plt.rc('font',family = 'Times New Roman')
        plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        plt.tick_params(axis = 'both',direction = 'in',labelsize = 15,width = 2)
        plt.tick_params(axis = 'both',which = 'minor',direction = 'in',labelsize = 15,width = 1.5,length = 4)
        plt.legend(fancybox=True, framealpha=0.01,fontsize = 16,frameon = False,loc = 'upper right')
        plt.xlabel(xlabel,size = 20)
        plt.ylabel(ylabel,size =20)
        plt.ylim = (-50,50)
        plt.title(title)
        if logx is True:
            plt.xscale('log')
        if logy is True:
            plt.yscale('log')
        if save is True:
            plt.savefig(FileName,dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
        if show is True:
            plt.show()
        plt.close()


    def trans_plot(self,yy, labels = ['Trans1','Trans2']):
        xx = (self.WavelengthArray13[50] + self.WavelengthArray24[50])/2
        myplot222.plot(xx,yy,xlabel = 'Neutron wavelength (A)', ylabel = 'Transmission', labels = labels, save = True, save_name = str("Data") +'_trans.png')

    def read_exp_data(self,FileName,StartLine):
        with open(FileName,"r") as f:
            Q = []
            IQ = []
            for line in f.readlines()[StartLine:]:
                sline = line.split()
                Q.append(float(sline[0]))
                IQ.append(float(sline[1]))
        Q = np.array(Q)
        IQ = np.array(IQ)
        return Q,IQ

DataFold2 = r'/data/zuotaisen/VSANS_data_reduce/data_reduce_system_new2/data_reduce_system_new2/NewStandardsD3TransDivR3'
DataFold3 = r'/data/zuotaisen/VSANS_data_reduce/data_reduce_system_new2/data_reduce_system_new2/NewStandardsDirectTransDivR3'
data_reduce = data_plot()
Q1,IQ1 = data_reduce.read_exp_data(DataFold2 + '/' +'Sample_TransPS-190K_2.2-6.7A_12.75m.dat',2)
Q2,IQ2 = data_reduce.read_exp_data(DataFold2 + '/' +'Sample_TransPS-190K_2.2-6.7A_12.75m.dat',2)
Q3,IQ3 = data_reduce.read_exp_data(DataFold3 + '/' +'Sample_TransPS-190K_2.2-6.7A_12.75m.dat',2)


data_reduce.plot_data([Q1,Q2,Q3],[IQ1,IQ2,IQ3],label = ['SANS2D','DivR3_D3Direct','DivR3_CenterDirect'], title = 'TransComparison', xlabel = 'Wavelength (Angstrom)',ylabel = 'Transmission',logx = False,logy = False, save = True,FileName = 'Comparison.png',show = True)
        
















