import numpy as np
from matplotlib import pyplot as plt
import os
import re
import sys

# 导入 instrument_info 模块
sys.path.insert(0, os.path.dirname(__file__))
from modules.instrument_reader import instrument_info


class data_plot():
    def __init__(self):
        pass
    
    def trim_zeros(self, x, y):
        """
        去除数据两端为0的点
        
        参数:
            x: x轴数据
            y: y轴数据
        
        返回:
            x_trimmed, y_trimmed: 去除两端0值后的数据
        """
        if len(y) == 0:
            return x, y
        
        # 找到第一个和最后一个非零值的索引
        nonzero_indices = np.where(y != 0)[0]
        
        if len(nonzero_indices) == 0:
            return x, y
        
        start_idx = nonzero_indices[0]
        end_idx = nonzero_indices[-1] + 1
        
        return x[start_idx:end_idx], y[start_idx:end_idx]

    def plot_data(self,xx,yy,label = np.arange(1,100),save = False, FileName = 'SavedPlot.png', title = 'title', xlabel = 'Lambda (Angstrom)', ylabel = 'Counts (n/s/Angstrom)',show = False, logx = False, logy = False, trim_zeros = True):        
        for i in range(len(xx)):#(len(tof)):
            if trim_zeros:
                x_trimmed, y_trimmed = self.trim_zeros(xx[i], yy[i])
                plt.plot(x_trimmed, y_trimmed, label=label[i])
            else:
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

    def plot_data_with_errorbar(self,xx,yy,error,label = np.arange(1,100),save = False, FileName = 'SavedPlot.png', title = 'title', xlabel = 'Lambda (Angstrom)', ylabel = 'Counts (n/s/Angstrom)', logx = False, logy = False, show = False, trim_zeros = True):        
        for i in range(len(xx)):#(len(tof)):
            if trim_zeros:
                x_trimmed, y_trimmed = self.trim_zeros(xx[i], yy[i])
                # 璁＄畻 trim 鍚庣殑绱㈠紩鑼冨洿
                start_idx = np.where(yy[i] != 0)[0][0] if len(np.where(yy[i] != 0)[0]) > 0 else 0
                end_idx = np.where(yy[i] != 0)[0][-1] + 1 if len(np.where(yy[i] != 0)[0]) > 0 else len(yy[i])
                error_trimmed = error[i][start_idx:end_idx]
                plt.errorbar(x_trimmed, y_trimmed, error_trimmed, label=label[i])
            else:
                plt.errorbar(xx[i],yy[i],error[i],label=label[i])
        plt.rcParams.update({'font.size': 15})
        plt.rc('font',family = 'Times New Roman')
        #plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
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
        #plt.close()



    def read_exp_data(self,FileName,StartLine):
        with open(FileName,"r") as f:
            Q = []
            IQ = []
            SigmaIQ = []
            SigmaQ = []
            for line in f.readlines()[StartLine:]:
                sline = line.split()
                Q.append(float(sline[0]))
                IQ.append(float(sline[1]))
                SigmaIQ.append(float(sline[2]))
                SigmaQ.append(float(sline[3]))
        Q = np.array(Q)
        IQ = np.array(IQ)
        SigmaIQ = np.array(SigmaIQ)
        SigmaQ = np.array(SigmaQ)
        return Q,IQ,SigmaIQ,SigmaQ
def weighted_average(matrix1, matrix2):
    """
    对两个一维矩阵进行加权平均计算
    
    规则：
    1. 对于每个位置i，计算 (matrix1[i]^2 + matrix2[i]^2) / (matrix1[i] + matrix2[i])
    2. 如果其中一个值为0，另一个不为0，结果为不为0的值
    3. 如果两个值都为0，结果为0
    
    参数:
        matrix1 (list): 第一个一维矩阵
        matrix2 (list): 第二个一维矩阵（长度必须与matrix1相同）
        
    返回:
        list: 加权平均结果矩阵
    """
    # 检查输入长度是否相同
    if len(matrix1) != len(matrix2):
        raise ValueError("两个矩阵的长度必须相同")
    
    result = []
    
    for i in range(len(matrix1)):
        a = matrix1[i]
        b = matrix2[i]
        
        # 处理特殊情况
        if a == 0 and b == 0:
            # 两个值都为0
            result.append(0.0)
        elif a == 0:
            # 第一个值为0，第二个值不为0
            result.append(float(b))
        elif b == 0:
            # 第二个值为0，第一个值不为0
            result.append(float(a))
        else:
            # 两个值都不为0，使用公式计算
            numerator = a*a + b*b
            denominator = a + b
            result.append(numerator / denominator)
    
    return result

def weighted_average_multi(*matrices):
    """
    对任意数量的矩阵进行加权平均计算
    
    参数:
        *matrices: 任意数量的一维矩阵（长度必须相同）
        
    返回:
        list: 加权平均结果矩阵
    """
    #if len(matrices) < 2:
    #    raise ValueError("至少需要两个矩阵")
    #n = len(matrices[0])
    #for i, mat in enumerate(matrices[1:], 1):
    #    if len(mat) != n:
    #        raise ValueError(f"矩阵{i}的长度({len(mat)})与第一个矩阵的长度({n})不同")

    result = []

    #for i in range(n):
    values0 = [mat for mat in matrices]
    dim1 = len(values0)
    dim2 = len(values0[1])
    values = np.zeros((dim1,dim2))
    values = np.array(values0)
#    for i in range(dim1):
#        print(values0[i])
#        for j in range(dim2):
#            values[i,j] = values0[i][j]
    numerator = np.sum(values**2,axis = 0) #sum(x*x for x in values)
    denominator = np.sum(values,axis = 0)
    #print(numerator)
    for t in range(len(numerator)):
        if abs(denominator[t]) < 1e-10:
            result.append(0.0)
        else:
            result.append(numerator[t] / denominator[t])
    return np.array(result)

def get_fold_name_and_instrument_info():
    # 从batchrun.py中读取FoldName - cmd开头的行中run.py的最后一个参数
    # 同时也读取instrument_info文件路径
    batchrun_path = os.path.join(os.path.dirname(__file__), 'batchrun.py')
    FoldName = None
    instrument_info_path = None
    
    with open(batchrun_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('cmd'):
                # 提取run.py后面的所有参数，找到最后一个带引号的参数
                # 格式: python run.py ... 'PS_spheres_SANS'
                match = re.search(r"run\.py\s+(.+)", line)
                if match:
                    params_str = match.group(1)
                    # 找到所有单引号包围的参数，取最后一个
                    all_matches = re.findall(r"'([^']+)'", params_str)
                    if all_matches:
                        FoldName = all_matches[-1]
                    
                    # 提取第一个参数（instrument_info文件路径）
                    param_list = params_str.split()
                    if param_list:
                        instrument_info_path = param_list[0]
                    break
    return FoldName, instrument_info_path


def get_wavelength_ranges(instrument_info_path):
    """从instrument_info文件中读取StartPoints和StopPoints"""
    # 实例化instrument_info对象
    instr = instrument_info(instrument_info_path)
    StartWave = instr.StartPoints
    StopWave = instr.StopPoints
    return StartWave, StopWave

def plot_all_sections(FoldName, StartWave, StopWave):

    # 获取波长段数
    n = len(StartWave)

    # 从FoldName文件夹中读取FileName - 第一个含有OnlySample的.dat文件名中提取OnlySample到.dat之间的部分
    # 注意：这里不再添加 './' 前缀，保持与原代码一致
    
    FileName = None
    FoldName2 = FoldName + '_0'
    FileNames = []
    if os.path.exists(FoldName2):
        files = os.listdir(FoldName2)
        for f in files:
            # 找到第一个包含'OnlySample'且以'.dat'结尾的文件
            if 'OnlySample' in f and f.endswith('.dat'):
                # 提取从'OnlySample'到'.dat'之间的部分
                # 例如: OnlyCell 从文件 IQ_Normed_with_QXErrorD1_OnlyCell_PSd8_2.2-6.7A_12.75m.dat
                match = re.search(r'OnlySample(.+?)\.dat', f)
                #print(match)
                if match:
                    FileNames.append('OnlySample' + match.group(1))
                    #break
    #FileName = FileNames[SampleNum]
    SaveFold = FoldName + FileNames[0]
    os.system('mkdir ' + SaveFold)
    
    for FileName in FileNames:
        # 如果FoldName是相对路径，确保路径正确
        if not os.path.isabs(FoldName):
            FoldName = r'./' + FoldName
        
        # 如果没找到，抛出异常或使用默认值
        if FileName is None:
            raise ValueError(f"在文件夹 {FoldName} 中找不到包含'Only'的.dat文件")
        
        # FileName已经是正确的值，不需要额外加单引号
        # (在文件路径拼接时会自动处理)
        
        if FoldName is None:
            raise ValueError("无法从batchrun.py中找到FoldName参数")
        
        data_reduce = data_plot()
        Qave = []
        IQave = []
        SigmaIQave = []
        for t in range(1,4):
            for i in range(t,t+1):
                for j in range(n):
                    fold_j = FoldName + '_' + str(j)
                    Q,IQ,SigmaIQ,SigmaQ = data_reduce.read_exp_data(fold_j + '/' +'IQ_Normed_with_QXErrorD' + str(i) + '_' + FileName + '.dat',2) 
                    DataFile = fold_j + '/' +'IQ_Normed_with_QXErrorD' + str(i) + '_' + FileName + '.dat'
                    os.system('cp ' + DataFile + ' ' +  SaveFold)
                    data_reduce.plot_data_with_errorbar([Q],[IQ],[SigmaIQ],label = ['D' + str(i) + '_' + str(StartWave[j]) + '-' + str(StopWave[j]) + ' Å'], title = 'Lambda divided', xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True,logy = True, save = False ,FileName = 'Comparison.png',show = False)
                    Qave.append(Q)
                    IQave.append(IQ)
                    SigmaIQave.append(SigmaIQ)
            #plt.show()
            plt.savefig(SaveFold + '/' + FileName + 'D' + str(t) + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
            plt.close()

        for t in range(3):
            #print(IQave)
            QAverage = weighted_average_multi(Qave[t*4],Qave[t*4+1],Qave[t*4+2],Qave[t*4+3])
            IQAverage = weighted_average_multi((IQave[t*4]),(IQave[t*4+1]),(IQave[t*4+2]),(IQave[t*4+3]))
            SigmaIQAverage = weighted_average_multi(SigmaIQave[t*4],SigmaIQave[t*4+1],SigmaIQave[t*4+2],SigmaIQave[t*4+3])
            #print(IQAverage)
            data_reduce.plot_data_with_errorbar([QAverage],[IQAverage],[SigmaIQAverage],label = ['D' + str(t+1) + '_average'], title = 'Lambda average', xlabel = 'Q (Å$^{-1}$)',ylabel = 'IQ (cm$^{-1}$)',logx = True,logy = True, save = False ,FileName = 'Comparison2.png',show = False)
            plt.savefig(SaveFold + '/' + FileName + 'Average_D' + str(t+1) + '.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
            plt.close()
        print('Complete ' + FileName + '   D' + str(t))
if __name__=="__main__":
    FoldName, instrument_info_path = get_fold_name_and_instrument_info()
    StartWave, StopWave = get_wavelength_ranges(instrument_info_path)
    plot_all_sections(FoldName, StartWave, StopWave)

