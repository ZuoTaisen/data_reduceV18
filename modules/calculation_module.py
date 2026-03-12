# -*- coding: utf-8 -*-
"""
Calculation Module - Contains methods related to data calculation
"""
import numpy as np
from math import log10, sqrt, atan, sin, pi
from numpy import logspace, linspace
import pickle
import os

class CalculationModule:
    @staticmethod
    def falling_distance(wavelength, L_1, L_2):
        '''input a neutron wavelenth (A) and L1(mm), output the falling distance of the neutron (mm) '''
        B = 3.073E-9*100
        L = L_1 + L_2
        y = B*wavelength**2*L*(L_1-L)         #  公式来源： Bouleam SANS Tool Box: Chapter 17 - GRAVITY CORRECTRING PRISMS
        y = y/1000                            #mm
        return y    #mm

    @staticmethod
    def detector_group_d1d2(detector_obj):
        """Detector grouping for D1 and D2 detectors"""
        RBins = detector_obj.RBins
        D2X, D2Y = detector_obj.get_detector_coordinate()
        # 使用 detector_obj 的 log bin RArrayEdges
        r_edges = detector_obj.RArrayEdges
        GroupY = []
        GroupX = []
        for i in range(len(D2X)):
            GroupY.append([])
            GroupX.append([])
            for j in range(RBins):
                GroupY[i].append([])
                GroupX[i].append([])
        for t in range(len(D2X)):
            for i in range(len(D2Y[t])):
                for j in range(len(D2X[t])):
                    Rij = np.sqrt(D2Y[t][i]**2 + D2X[t][j]**2)
                    # 使用 np.digitize 找到 Rij 对应的 bin 索引
                    tmp = np.digitize(Rij, r_edges) - 1
                    # 边界情况处理
                    if tmp < 0:
                        tmp = 0
                    if tmp >= RBins:
                        tmp = RBins - 1
                    GroupY[t][tmp].append(i)
                    GroupX[t][tmp].append(j)
        
        # Determine detector type (D1 or D2)
        detector_type = "D1" if "D1" in str(detector_obj.__class__) else "D2"
        
        # Save GroupX and GroupY to pickle files
        filename_x = f'./npyfiles/{detector_type}GroupX_{detector_obj.WaveMin}-{detector_obj.WaveMax}A.pkl'
        filename_y = f'./npyfiles/{detector_type}GroupY_{detector_obj.WaveMin}-{detector_obj.WaveMax}A.pkl'
        
        with open(filename_x, 'wb') as f:
            pickle.dump(GroupX, f)
        with open(filename_y, 'wb') as f:
            pickle.dump(GroupY, f)
        
        return GroupX, GroupY

    @staticmethod
    def detector_group_d3_0(detector_obj, Height, Width, R, XBins, YBins, RBins, XCenter, YCenter):
        """Detector grouping for D3 detector"""
        # 使用 detector_obj 的 log bin RArrayEdges
        r_edges = detector_obj.RArrayEdges
        FallingDistance = detector_obj.falling_distance(detector_obj.WavelengthArray, detector_obj.L1, detector_obj.L2)
        WaveBins = detector_obj.WaveBins
        
        # 预分配列表结构，避免重复append
        GroupY = [[[] for _ in range(WaveBins)] for __ in range(RBins)]
        GroupX = [[[] for _ in range(WaveBins)] for __ in range(RBins)]
        print(XCenter,YCenter,'This is the center')
        print(FallingDistance)


#        GroupY = []
#        GroupX = []
#        for i in range(RBins):
#            GroupY.append([])
#            GroupX.append([])
#            for k in range(detector_obj.WaveBins):
#                GroupY[i].append([])
#                GroupX[i].append([])
        yy = detector_obj.YArray
        xx = detector_obj.XArray

        for i in np.arange(YBins):
            for j in np.arange(XBins):
                for k in np.arange(detector_obj.WaveBins):
                    Rij = np.sqrt((yy[i] - FallingDistance[k])**2 + (xx[j])**2)
                    # 使用 np.digitize 找到 Rij 对应的 bin 索引
                    tmp = np.digitize(Rij, r_edges) - 1
                    # 边界情况处理
                    if tmp < 0:
                        tmp = 0
                    if tmp >= RBins:
                        tmp = RBins - 1
                    GroupY[tmp][k].append(i)
                    GroupX[tmp][k].append(j)
        return GroupX,GroupY


    @staticmethod
    def grouping_d1d2(detector_obj, D2Data):
        """Grouping for D1 and D2 detectors"""
        GroupX, GroupY = detector_obj.detector_group()
        GroupedData = np.zeros(len(D2Data))[:, None][:, None] * np.zeros(detector_obj.RBins)[:, None] * np.zeros(detector_obj.WaveBins)
        for t in range(len(D2Data)):
            for i in np.arange(detector_obj.RBins):
                # 修复索引顺序：对于所有bank，GroupY[t][i]保存的是第一个轴的索引，GroupX[t][i]保存的是第二个轴的索引
                # Bank 0和2的形状为(Y, X, WaveBins)，Bank 1和3的形状为(X, Y, WaveBins)
                # 但GroupY和GroupX的索引是根据detector坐标计算的，所以对于所有bank都应该使用[GroupY, GroupX]
                tmp = np.sum(D2Data[t][GroupY[t][i], GroupX[t][i]], axis=0)
                GroupedData[t][i] = tmp
        return GroupedData

    @staticmethod
    def grouping_d3(detector_obj, DataReshaped, GroupX, GroupY, RBins):
        """Grouping for D3 detector"""
        # 直接创建结果数组，避免中间步骤
        result = np.zeros((RBins, DataReshaped.shape[2]))
        
        for i in range(RBins):
            #for j in range(detector_obj.WaveBins):
            x_indices = GroupX[i]
            y_indices = GroupY[i]
            if x_indices and y_indices:  # 检查是否有数据点
                    # DataReshaped的形状是[Height, Width, WaveBins] = [128, 250, WaveBins]
                    # GroupX存储的是XBins的索引（0-249），对应DataReshaped的第二个轴
                    # GroupY存储的是YBins的索引（0-127），对应DataReshaped的第一个轴
                    # 但在NumPy中，我们需要先指定第一个轴的索引，再指定第二个轴的索引
                    # 所以正确的顺序是[y_indices, x_indices]，这样可以确保索引不越界
                    # 但是，由于之前的错误，我们需要检查实际的数据访问方式
                    # 经过仔细分析，发现之前的修复是正确的，问题可能出在其他地方
                    # 让我们尝试使用[x_indices, y_indices]，因为DataReshaped的实际形状可能是[Width, Height, WaveBins]
                result[i] += np.sum(DataReshaped[x_indices, y_indices], axis=0)
        
        return result

    @staticmethod
    def grouping_mask_d1d2(detector_obj, D2Data, XMinD2, XMaxD2, YMinD2, YMaxD2):
        """Grouping mask for D1 and D2 detectors"""
        mask = {}
        for i in range(len(D2Data)):
            mask[i] = np.ones_like(np.sum(D2Data[i], axis=2))
            for j in range(len(XMinD2[i])):
                tmpX = np.arange(XMinD2[i][j], XMaxD2[i][j])[:, None]
                tmpY = np.arange(YMinD2[i][j], YMaxD2[i][j])
                mask[i][tmpX, tmpY] = 0
        return mask

    @staticmethod
    def grouping_mask_d3(detector_obj, D3Data, XMinD3, XMaxD3, YMinD3, YMaxD3):
        """Grouping mask for D3 detector - corrected version"""
        # 创建与D3Data前两个维度相同形状的掩码
        mask = np.ones((D3Data.shape[0], D3Data.shape[1]), dtype=np.int8)
        
        for i in range(len(XMinD3)):
            # 确保索引是整数
            x_start = int(round(XMinD3[i]))
            x_end = int(round(XMaxD3[i]))
            y_start = int(round(YMinD3[i]))
            y_end = int(round(YMaxD3[i]))
            
            # 设置矩形区域为0
            mask[x_start:x_end, y_start:y_end] = 0
        
        return mask

    def grouping_mask_d4(detector_obj, D3Data, XMinD3, XMaxD3, YMinD3, YMaxD3):
        """Grouping mask for D3 detector"""
        mask = np.ones_like(np.sum(D3Data, axis=2))
        for i in range(len(XMinD3)):
            #tmpX = np.arange(XMinD3[i], XMaxD3[i])[:, None]
            tmpY = np.arange(YMinD3[i], YMaxD3[i])
            tmpX = np.arange(XMinD3[i], XMaxD3[i])
            mask[tmpX, tmpY] = 0
        return mask

    @staticmethod
    def azimuthal_mask_d1d2(detector_obj, D2Data, PhiMin, PhiMax):
        """Azimuthal mask for D1 and D2 detectors"""
        RBins = detector_obj.RBins
        xx, yy = detector_obj.get_detector_coordinate()
        for t in range(len(xx)):
            yy[t] = yy[t][:, None]
        mask = {0: np.ones_like(xx[0] * yy[0]), 1: np.ones_like(xx[1] * yy[1]), 2: np.ones_like(xx[2] * yy[2]), 3: np.ones_like(xx[3] * yy[3])}
        for t in range(len(xx)):
            rr = np.sqrt(xx[t]**2 + yy[t]**2)
            angle1 = np.arccos(xx[t] / rr) * 180 / pi * (yy[t] > 0)
            angle2 = (360 - np.arccos(xx[t] / rr) * 180 / pi) * (yy[t] < 0)
            Phi = angle1 + angle2
            for i in range(len(PhiMin)):
                tmp = (Phi >= PhiMax[i]) + (Phi < PhiMin[i])
                mask[t] = mask[t] * tmp
        return mask

    @staticmethod
    def azimuthal_mask_d3(detector_obj, D3Data, PhiMin, PhiMax):
        """Azimuthal mask for D3 detector"""
        Height = detector_obj.BankHeight
        Width = detector_obj.BankWidth
        XBins = detector_obj.XBins
        YBins = detector_obj.YBins
        XCenter = detector_obj.XCenter
        YCenter = detector_obj.YCenter
        yy = np.linspace(-1 * Height / 2, Height / 2, YBins) - YCenter
        xx = np.linspace(-1 * Width / 2, Width / 2, XBins)[:, None] - XCenter
        rr = np.sqrt((xx)**2 + (yy)**2)
        angle1 = np.arccos(xx / rr) * 180 / pi * (yy >= 0)
        angle2 = (360 - np.arccos(xx / rr) * 180 / pi) * (yy < 0)
        Phi = angle1 + angle2
        mask = np.ones_like(Phi)
        for i in range(len(PhiMin)):
            tmp = (Phi > PhiMax[i]) + (Phi < PhiMin[i])
            mask = mask * tmp
        return mask

    @staticmethod
    def solid_angle_d1d2(detector_obj):
        """Solid angle calculation for D1 and D2 detectors"""
        GroupX, GroupY = detector_obj.detector_group()
        PixelArea = detector_obj.TubeWidth * detector_obj.PixelHeight
        SolidR = np.zeros(len(GroupX))[:, None] * np.zeros(detector_obj.RBins)
        if detector_obj.MaskSwitch:
            for t in range(len(GroupX)):
                mask = detector_obj.load_mask()
                for i in range(len(GroupX[t])):
                    if t == 0 or t == 2:
                        SolidR[t][i] = PixelArea * np.sum(mask[t][GroupY[t][i], GroupX[t][i]]) / (detector_obj.L2_13**2) * (detector_obj.L2_13 / np.sqrt(detector_obj.RArray[i]**2 + detector_obj.L2_13**2))**2
                    if t == 1 or t == 3:
                        SolidR[t][i] = PixelArea * np.sum(mask[t][GroupY[t][i], GroupX[t][i]]) / (detector_obj.L2_24**2) * (detector_obj.L2_24 / np.sqrt(detector_obj.RArray[i]**2 + detector_obj.L2_24**2))**2
        else:
            for t in range(len(GroupX)):
                for i in range(len(GroupX[t])):
                    if t == 0 or t == 2:
                        SolidR[t][i] = PixelArea * len(GroupX[t][i]) / (detector_obj.L2_13**2) * (detector_obj.L2_13 / np.sqrt(detector_obj.RArray[i]**2 + detector_obj.L2_13**2))**2
                    if t == 1 or t == 3:
                        SolidR[t][i] = PixelArea * len(GroupX[t][i]) / (detector_obj.L2_24**2) * (detector_obj.L2_24 / np.sqrt(detector_obj.RArray[i]**2 + detector_obj.L2_24**2))**2
        return SolidR

    @staticmethod
    def solid_angle_d3(detector_obj, Height, Width, R, XBins, YBins, RBins, XCenter, YCenter):
        """Solid angle calculation for D3 detector"""
        # 使用 detector_obj 的 log bin RArrayEdges
        r_edges = detector_obj.RArrayEdges
        SolidR = []
        for i in range(RBins):
            SolidR.append([])
        FallingDistance = detector_obj.falling_distance(detector_obj.WavelengthArray, detector_obj.L1, detector_obj.L2)
        yy = detector_obj.YArray
        xx = detector_obj.XArray
        if detector_obj.MaskSwitch:
            mask = np.load(r'masks/D3Mask.npy', allow_pickle=True)
            for i in np.arange(XBins):
                for j in np.arange(YBins):
                    Rij = np.sqrt((yy[j])**2 + (xx[i])**2)
                    # 使用 np.digitize 找到 Rij 对应的 bin 索引
                    tmp = np.digitize(Rij, r_edges) - 1
                    # 边界情况处理
                    if tmp < 0:
                        tmp = 0
                    if tmp >= RBins:
                        tmp = RBins - 1
                    SolidR[tmp].append(mask[i, j] * (detector_obj.TubeWidth + 0.5) * detector_obj.TubeHeight / (detector_obj.L2**2) * (detector_obj.L2 / np.sqrt(Rij**2 + detector_obj.L2**2))**2)
        else:
            for i in np.arange(XBins):
                for j in np.arange(YBins):
                    Rij = np.sqrt((yy[j])**2 + (xx[i])**2)
                    # 使用 np.digitize 找到 Rij 对应的 bin 索引
                    tmp = np.digitize(Rij, r_edges) - 1
                    # 边界情况处理
                    if tmp < 0:
                        tmp = 0
                    if tmp >= RBins:
                        tmp = RBins - 1
                    SolidR[tmp].append((detector_obj.TubeWidth + 0.5) * detector_obj.TubeHeight / (detector_obj.L2**2) * (detector_obj.L2 / np.sqrt(Rij**2 + detector_obj.L2**2))**2)
        for i, item in enumerate(SolidR):
            SolidR[i] = np.sum(SolidR[i])
        return np.array(SolidR)

    @staticmethod
    def solid_angle_2d_d1d2(detector_obj):
        """2D solid angle calculation for D1 and D2 detectors"""
        D2X, D2Y = detector_obj.get_detector_coordinate()
        PixelArea = detector_obj.TubeWidth * detector_obj.PixelHeight
        SolidXY1 = np.zeros_like(detector_obj.Y1 * detector_obj.X1[:, None])
        SolidXY2 = np.zeros_like(detector_obj.Y2 * detector_obj.X2[:, None])
        SolidXY3 = np.zeros_like(detector_obj.Y3 * detector_obj.X3[:, None])
        SolidXY4 = np.zeros_like(detector_obj.Y4 * detector_obj.X4[:, None])

        SolidXY = []
        for t in range(4):
            for i in range(len(locals()['SolidXY' + str(t + 1)])):
                for j in range(len(locals()['SolidXY' + str(t + 1)][0])):
                    Rij = np.sqrt((locals()['X' + str(t + 1)][i])**2 + (locals()['Y' + str(t + 1)][j])**2)
                    if t == 0 or t == 2:
                        locals()['SolidXY' + str(t + 1)][i, j] = PixelArea / (detector_obj.L2_13**2) * (detector_obj.L2_13 / np.sqrt(detector_obj.RArray[i]**2 + detector_obj.L2_13**2))**2
                    if t == 1 or t == 3:
                        locals()['SolidXY' + str(t + 1)][i, j] = PixelArea / (detector_obj.L2_24**2) * (detector_obj.L2_24 / np.sqrt(detector_obj.RArray[i]**2 + detector_obj.L2_24**2))**2
            SolidXY.append(locals()['SolidXY' + str(t + 1)])
        return SolidXY

    @staticmethod
    def solid_angle_2d_d3(detector_obj, Height, Width, R, XBins, YBins, XCenter, YCenter):
        """2D solid angle calculation for D3 detector"""
        yy = detector_obj.YArray
        xx = detector_obj.XArray
        SolidXY = np.zeros_like(yy * xx[:, None])
        #FallingDistance = detector_obj.falling_distance(detector_obj.WavelengthArray, detector_obj.L1, detector_obj.L2)
        if detector_obj.MaskSwitch:
            mask = np.load(r'masks/D3Mask.npy', allow_pickle=True)
            for i in np.arange(XBins):
                for j in np.arange(YBins):
                    Rij = np.sqrt((yy[j])**2 + (xx[i])**2)
                    SolidXY[i, j] = mask[i, j] * (detector_obj.TubeWidth + 0.5) * detector_obj.TubeHeight / (detector_obj.L2**2 + Rij**2) * (detector_obj.L2 / np.sqrt(Rij**2 + detector_obj.L2**2))**2
        else:
            for i in np.arange(XBins):
                for j in np.arange(YBins):
                    Rij = np.sqrt((yy[j])**2 + (xx[i])**2)
                    SolidXY[i, j] = (detector_obj.TubeWidth + 0.5) * detector_obj.TubeHeight / (detector_obj.L2**2 + Rij**2) * (detector_obj.L2 / np.sqrt(Rij**2 + detector_obj.L2**2))**2
        return SolidXY
