# This is to do rotation of 3D Magnetic Domains from the reconstruction
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

# transform the 3D reconstructed magnetic domains by (i) rotation around y-axis, followed by (ii) rotation around x-axis
# The function forwardtransform() is called from check_registration() function
def forwardtransform(arr, tform):
    for m in range(arr.shape[0]):
        arr[:, m, :] = tf.rotate(arr[:, m, :], tform[0], mode='edge')
    for m in range(arr.shape[2]):
        arr[m, :, :] = tf.rotate(arr[m, :, :], tform[1], mode='edge')

    arr = ndimage.shift(arr, (tform[2], tform[3], tform[4]), mode='constant', cval=0)
    return arr


# actual code for rotating the actual 3D object as well as the magnetic domains
def check_registration():
    
    # this angle has been obtained through trial and error
    ii = 32
    print('angle:', ii)

    # Load the NanoMAX reconstructions
    NM_recx_o = dxchange.read_tiff('tmp4/recx-9.tiff')
    NM_recy_o = dxchange.read_tiff('tmp4/recy-9.tiff')
    NM_recz_o = dxchange.read_tiff('tmp4/recz-9.tiff')

    print(NM_recx_o.shape, NM_recy_o.shape, NM_recz_o.shape)

    NM_recx = NM_recx_o.copy(); NM_recy = NM_recy_o.copy(); NM_recz = NM_recz_o.copy()
    
    # Rotate the 3D object by 180 degrees
    tform = np.array([180, -33.0, 0, 0, 0])
    
    # rotate the 3D object consisting of the magnetic domains
    NM_recx_r = forwardtransform(NM_recx, tform)
    NM_recy_r = forwardtransform(NM_recy, tform)
    NM_recz_r = forwardtransform(NM_recz, tform)

    #rotate just the magnetic domains around the x-axis to align the magnetic domains in the direction of maximum contrast
    rotmat = np.zeros((3, 3), dtype='float32')
    theta_rot = ii * np.pi / 180.
    rotmat[0, 0] = 1
    rotmat[0, 1] = 0
    rotmat[0, 2] = 0
    rotmat[1, 0] = 0
    rotmat[2, 0] = 0
    rotmat[1, 1] = np.cos(theta_rot)
    rotmat[1, 2] = -np.sin(theta_rot)
    rotmat[2, 1] = np.sin(theta_rot)
    rotmat[2, 2] = np.cos(theta_rot)
    mag_rot = np.dot(rotmat, [NM_recx_r.flatten(), NM_recy_r.flatten(), NM_recz_r.flatten()])

    # reshape to match the actual shape of the 3D magnetization domains.
    mx_rot = np.reshape(mag_rot[0], NM_recx_r.shape)
    my_rot = np.reshape(mag_rot[1], NM_recy_r.shape)
    mz_rot = np.reshape(mag_rot[2], NM_recz_r.shape)

    # this is the normalization step. The magnitude is first computed at each location from the magnetization components along x-axis, y-axis and z-axis.
    m_tot = np.zeros(NM_recx_r.shape)
    m_tot = np.sqrt(np.square(mx_rot) + np.square(my_rot) + np.square(mz_rot))
    print(np.max(m_tot), np.min(m_tot))

    # each magnetization component is divided by the magnitude to obtain the normalized magnetization component.
    mxn1 = np.divide(mx_rot, m_tot)
    myn1 = np.divide(my_rot, m_tot)
    mzn1 = np.divide(mz_rot, m_tot)
    
    # saving in an array.
    folder_pa = 'rotated' + str(ii) 
    dxchange.write_tiff(mxn1, folder_pa + '/recy', dtype='float32', overwrite=True)
    dxchange.write_tiff(myn1, folder_pa + '/recz', dtype='float32', overwrite=True)
    dxchange.write_tiff(mzn1, folder_pa + '/recx', dtype='float32', overwrite=True)
    

if __name__ == "__main__":
    check_registration()