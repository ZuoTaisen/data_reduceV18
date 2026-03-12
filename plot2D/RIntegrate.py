import sys
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def radial_integration(QX, QY, QArray, QR1, QR2, num_bins=40):
    # 计算每个点的极坐标
    R = np.sqrt(QX**2 + QY**2)  # 计算半径
    Theta = np.arctan2(QY, QX)  # 计算方位角

    # 选择在径向环形区域 QR1 到 QR2 之间的点
    mask = (R >= QR1) & (R <= QR2)  # 判断哪些点在指定的径向区域内

    # 获取这些点的方位角和对应的QArray值
    theta_values = Theta[mask]
    qarray_values = QArray[mask]

    # 创建方位角的区间
    theta_bins = np.linspace(-np.pi, np.pi, num_bins+1)  # 方位角从 -pi 到 pi，分成 num_bins 个区间

    # 计算每个方位角区间内的积分
    integral_values = np.zeros(num_bins)
    
    for i in range(num_bins):
        # 找到每个区间内的方位角
        mask_theta = (theta_values >= theta_bins[i]) & (theta_values < theta_bins[i+1])
        # 对这些点的QArray值进行积分
        integral_values[i] = np.sum(qarray_values[mask_theta])

    # 返回每个区间的积分值和对应的方位角中点
    theta_midpoints = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    return integral_values, theta_midpoints


RunNum = sys.argv[1]
QX0 = RunNum + 'QX.txt.npy'
QY0 = RunNum + 'QY.txt.npy'
QArray0 = RunNum + 'QArray.npy'
QX = np.load(QX0)
QY = np.load(QY0)
QArray = np.load(QArray0)
#radial_integration(QX, QY, QArray, 0.02,0.04)


# 示例：假设我们有一个矩阵 QArray 和相应的 QX, QY 坐标
# 生成一些示例数据
#x = np.linspace(-10, 10, 100)
#y = np.linspace(-10, 10, 100)
QX, QY = np.meshgrid(QX,QY)
#QArray = np.exp(-0.1 * (QX**2 + QY**2))  # 假设 QArray 是一个高斯分布的二维函数

# 定义径向环形区域的范围
QR1 = 0.01  # 内半径
QR2 = 0.05  # 外半径

# 调用径向积分函数
integral_value, theta_values = radial_integration(QX, QY, QArray, QR1, QR2,100)

# 打印积分结果
print("径向积分值:", integral_value)
print("方位角theta的值范围:", np.min(theta_values), "到", np.max(theta_values))

# 可视化方位角的分布
#plt.hist(theta_values, bins=360, edgecolor='black')
plt.plot(theta_values*180/np.pi,integral_value)
plt.title("Distribution of Theta values in the radial region")
plt.xlabel("Theta (radians)")
plt.ylabel("Frequency")
plt.show()

