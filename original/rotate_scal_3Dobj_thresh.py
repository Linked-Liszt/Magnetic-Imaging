# This is to do 3D Vector Tomographic Reconstruction from Ptychographic Projections
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import tomopy
import dxchange
from scipy import signal, ndimage, misc
from skimage import transform as tf
from skimage.draw import disk
import time
from scipy.spatial.transform import Rotation as R


def forwardtransform(arr, tform):
    for m in range(arr.shape[0]):
        arr[:, m, :] = tf.rotate(arr[:, m, :], tform[0], mode='edge')
    for m in range(arr.shape[2]):
        arr[m, :, :] = tf.rotate(arr[m, :, :], tform[1], mode='edge')

    arr = ndimage.shift(arr, (tform[2], tform[3], tform[4]), mode='constant', cval=0)
    return arr

def check_registration():
    # Load just the rec1_sirt reconstruction
    NM_scal_gr = dxchange.read_tiff('rec1_sirt.tiff')
    NM_scal_gr1 = NM_scal_gr.copy()
    
    # Rotate the 3D scalar magnetization by the desired rotation angles around x-axis and y-axis
    tform = np.array([180, -33.0, 0, 0, 0])
    NM_scal_gr1a = forwardtransform(NM_scal_gr1, tform)
    
    # save the rotated file
    file_pa = 'rotated_SIRT/rec1_sirt_rot'  
    dxchange.write_tiff(NM_scal_gr1a, file_pa, dtype='float32', overwrite=True)

    # the vector object is just extracted out
    mx[NM_scal_gr1a < 0.009] = 0
    my[NM_scal_gr1a < 0.009] = 0
    mz[NM_scal_gr1a < 0.009] = 0

    # save the vector field only
    file_pa = '3D_mag/'  
    dxchange.write_tiff(mx, file_pa + 'mx_samp_only', overwrite=True)
    dxchange.write_tiff(my, file_pa + 'my_samp_only', overwrite=True)
    dxchange.write_tiff(mz, file_pa + 'mz_samp_only', overwrite=True)


if __name__ == "__main__":
    check_registration()