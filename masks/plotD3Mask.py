import numpy as np

DataD3 = np.load('D3Data.npy')
DataD3Mask = np.load('D3Mask.npy')
import matplotlib
from matplotlib import pyplot as plt

#print(DataD3[DataD3>1])
plt.figure(1)
ax = plt.matshow(np.log10(np.transpose(DataD3)),cmap = plt.cm.jet)
#plt.colorbar(ax.colorbar,fraction = 1000)
plt.title("2D map")
plt.colorbar()#.set_label(Data2D)
plt.xlabel('Horizontal dimension(Pixels)')
plt.ylabel('Vertical diemension(Pixels)')
plt.savefig('OutputD3_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
#plt.show()
#plt.close()
plt.figure(2)
ax = plt.matshow(np.log10(np.transpose(DataD3*DataD3Mask)),cmap = plt.cm.jet)
#plt.colorbar(ax.colorbar,fraction = 1000)
plt.title("2D map")
plt.colorbar()#.set_label(Data2D)
plt.xlabel('Horizontal dimension(Pixels)')
plt.ylabel('Vertical diemension(Pixels)')
plt.savefig('OutputD3_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
plt.show()
