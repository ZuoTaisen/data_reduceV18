import numpy as np
import matplotlib
from matplotlib import pyplot as plt

for i in range(4):
    DataD2 = np.load('D1Data'+ str(i) +'.npy')
    MaskD2 = np.load('D1Mask'+ str(i) +'.npy')
    print('D1' + str(i+1) + ' :')
    ax = plt.matshow((DataD2),cmap = plt.cm.jet)
    #plt.colorbar(ax.colorbar,fraction = 1000)
    plt.title("2D map D1" + str(i))
    plt.colorbar()#.set_label(Data2D)
    plt.ylabel('Vertical dimension(Pixels)')
    plt.xlabel('Horizontal diemension(Pixels)')
    plt.savefig('OutputD3_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()
    plt.close()

    ax = plt.matshow((DataD2*MaskD2),cmap = plt.cm.jet)
    plt.title("2D map D1" + str(i))
    plt.colorbar()#.set_label(Data2D)
    plt.ylabel('Vertical dimension(Pixels)')
    plt.xlabel('Horizontal diemension(Pixels)')
    plt.savefig('OutputD3_2D.png',dpi = 600, format = 'png',bbox_inches = 'tight', transparent = True)
    plt.show()

