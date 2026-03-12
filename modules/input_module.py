# -*- coding: utf-8 -*-
"""
Input Module - Contains methods related to data input and loading
"""
import numpy as np
import re
import h5py
from datetime import datetime

class InputModule:
    @staticmethod
    def get_proton_charge(file):
        f = open(file, 'r')
        txt = f.readlines()
        pattern = re.compile(r'(\d+\.?\d*)</proton_charge>?')
        tmp = pattern.findall(str(txt))
        ProtonCharge = float(tmp[0])
        return ProtonCharge

    @staticmethod
    def get_now():
        # 获取当前的日期和时间
        current_datetime = datetime.now()
        
        # 按照指定的格式输出
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # 打印结果
        return formatted_datetime

    @staticmethod
    def time_diff(time1, time2):
        # 将字节格式转换为字符串
        time1_str = time1.decode('utf-8')
        time2_str = time2.decode('utf-8')

        # 定义时间格式
        time_format = '%Y-%m-%d %H:%M:%S'

        # 将字符串时间转换为 datetime 对象
        time1_obj = datetime.strptime(time1_str, time_format)
        time2_obj = datetime.strptime(time2_str, time_format)

        # 计算时间差
        time_difference = time2_obj - time1_obj

        # 计算差值的秒数和分钟数
        difference_seconds = time_difference.total_seconds()
        difference_minutes = difference_seconds / 60

        return difference_seconds, difference_minutes

    @staticmethod
    def load_mask(mask_type, num_banks=4):
        """Load mask for different detector types (D1, D2, D3)"""
        mask = []
        if mask_type in ['D1', 'D2']:
            for i in range(num_banks):
                tmp = np.load('masks/' + mask_type + 'Mask' + str(i) + '.npy', allow_pickle=True)
                mask.append(tmp)
        elif mask_type == 'D3':
            mask = np.load('masks/D3Mask.npy', allow_pickle=True)
        return mask