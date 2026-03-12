import numpy as np
import re
from datetime import datetime
def get_experimental_time(DataFold,RunNum):
    DataFileName = DataFold + "/" + str(RunNum) + "/" + "detector.nxs"
    #RunInfoFileName = glob.glob(self.RunInfoFold + '/' + str(RunNum) + '/' + '**.xml')[0]
    f = h5py.File(DataFileName, "r")
    startTime = f["/csns/start_time_utc"][()][0]
    endTime = f["/csns/end_time_utc"][()][0]
    useTime = self.time_diff(startTime,endTime)
    useTimeMin = useTime[1]
    OutPut = [startTime,endTime,useTime,useTimeMin]
    return OutPut

def get_experimental_time_info(DataFold,RunNum):
    FileName = DataFold + "/" + str(RunNum) + "/" + str(RunNum)
    with open(FileName, 'r') as f:
        content = f.read()

    # 提取时间字符串
    start_time = re.search(r'"start_time_utc":\s*"([^"]+)"', content).group(1)
    end_time = re.search(r'"end_time_utc":\s*"([^"]+)"', content).group(1)

    # 计算时间差
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    time_diff_min = (end - start).total_seconds() / 60
    time_diff_sec = (end - start).total_seconds()
    return [start,end,time_diff_sec,time_diff_min]

