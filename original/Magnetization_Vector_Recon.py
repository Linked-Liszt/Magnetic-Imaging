'''Python code for Magnetization Vector Reconstruction from STXM-Tomography Experimental Data at NanoMAX'''
import tomopy
import dxchange 
import numpy as np
from skimage import transform
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 9, 'font.family' : 'times'})
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial.transform import Rotation as R


'''
The tomographic projections are first prepared by normalizing the background and performing minus_log operations. 

In this example, 0002_tomogram1 and 0004_tomogram2 are the 2D STXM projections for the 2.1 micron NFB cylinder 
corresponding to the left cicrular (CL) and right circular (CR) polarized X-rays respectively.

On the other hand, 0007_tomogram3 and 0008_tomogram4 are the 2D STXM projections for the 5.4 micron NFB cylinder
corresponding to the left circular (CL) and right circular (CR) polarized X-rays respectively. The pixel values are also 
appropriately scaled.

The projections are appropriately padded using np.pad() to make the array size same.

# 



'''




def prep_projs(base, proc):
    # 0002_tomogram1
    prjl = np.load(base + '0002_tomogram1-stxm_cl-150-cor.npy')
    prjr = np.load(base + '0002_tomogram1-stxm_cr-150-cor.npy')
    ang = np.load(base + '0002_tomogram1-ang-150.npy')

    #load 1

    dxchange.write_tiff(prjl, proc + 'raw/tif/0002_tomogram1-stxm_cl-150-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'raw/tif/0002_tomogram1-stxm_cr-150-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'raw/tif/0002_tomogram1-ang-150-cor', overwrite=True)
    np.save(proc + 'raw/npy/0002_tomogram1-stxm_cl-150-cor.npy', prjl)
    np.save(proc + 'raw/npy/0002_tomogram1-stxm_cr-150-cor.npy', prjr)
    np.save(proc + 'raw/npy/0002_tomogram1-ang-150-cor.npy', ang)

    # save 1

    prjl = tomopy.normalize_bg(prjl)
    prjl = tomopy.minus_log(prjl)
    prjr = tomopy.normalize_bg(prjr)
    prjr = tomopy.minus_log(prjr)
    npad = ((0, 0), (0, 0), (40, 40))
    prjl = np.pad(prjl, npad, mode='constant', constant_values=0)
    prjr = np.pad(prjr, npad, mode='constant', constant_values=0)

    # norm pad 1

    dxchange.write_tiff(prjl, proc + 'prepared/tif/0002_tomogram1-stxm_cl-150-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'prepared/tif/0002_tomogram1-stxm_cr-150-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'prepared/tif/0002_tomogram1-ang-150-cor', overwrite=True)
    np.save(proc + 'prepared/npy/0002_tomogram1-stxm_cl-150-cor.npy', prjl)
    np.save(proc + 'prepared/npy/0002_tomogram1-stxm_cr-150-cor.npy', prjr)
    np.save(proc + 'prepared/npy/0002_tomogram1-ang-150-cor.npy', ang)

    # write 1

    # 0004_tomogram2
    prjl = np.load(base + '0004_tomogram2-stxm_cl-50-cor.npy')
    prjr = np.load(base + '0004_tomogram2-stxm_cr-50-cor.npy')
    ang = np.load(base + '0004_tomogram2-ang-50.npy')

    #load 2 
    dxchange.write_tiff(prjl, proc + 'raw/tif/0004_tomogram2-stxm_cl-50-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'raw/tif/0004_tomogram2-stxm_cr-50-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'raw/tif/0004_tomogram2-ang-50-cor', overwrite=True)
    np.save(proc + 'raw/npy/0004_tomogram2-stxm_cl-50-cor.npy', prjl)
    np.save(proc + 'raw/npy/0004_tomogram2-stxm_cr-50-cor.npy', prjr)
    np.save(proc + 'raw/npy/0004_tomogram2-ang-50-cor.npy', ang)
    prjl = tomopy.normalize_bg(prjl)
    prjl = tomopy.minus_log(prjl)
    prjr = tomopy.normalize_bg(prjr)
    prjr = tomopy.minus_log(prjr)
    dxchange.write_tiff(prjl, proc + 'prepared/tif/0004_tomogram2-stxm_cl-50-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'prepared/tif/0004_tomogram2-stxm_cr-50-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'prepared/tif/0004_tomogram2-ang-50-cor', overwrite=True)
    np.save(proc + 'prepared/npy/0004_tomogram2-stxm_cl-50-cor.npy', prjl)
    np.save(proc + 'prepared/npy/0004_tomogram2-stxm_cr-50-cor.npy', prjr)
    np.save(proc + 'prepared/npy/0004_tomogram2-ang-50-cor.npy', ang)

    # 0007_tomogram3 
    prjl = np.load(base + '0007_tomogram3-stxm_cl-135.npy')
    prjr = np.load(base + '0007_tomogram3-stxm_cr-135.npy')
    ang = np.load(base + '0007_tomogram3-ang-135.npy')
    #load 3
    dxchange.write_tiff(prjl, proc + 'raw/tif/0007_tomogram3-stxm_cl-135-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'raw/tif/0007_tomogram3-stxm_cr-135-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'raw/tif/0007_tomogram3-ang-135-cor', overwrite=True)
    np.save(proc + 'raw/npy/0007_tomogram3-stxm_cl-135-cor.npy', prjl)
    np.save(proc + 'raw/npy/0007_tomogram3-stxm_cr-135-cor.npy', prjr)
    np.save(proc + 'raw/npy/0007_tomogram3-ang-135-cor.npy', ang)
    #save

    scl = np.max(prjl[0])
    prjl = prjl / scl
    prjr = prjr / scl
    prjl[prjl > 1] = 0.995
    prjr[prjr > 1] = 0.995
    prjl = tomopy.normalize_bg(prjl)
    prjl = tomopy.minus_log(prjl)
    prjr = tomopy.normalize_bg(prjr)
    prjr = tomopy.minus_log(prjr)
    npad = ((0, 0), (10, 0), (15, 15))
    prjl = np.pad(prjl, npad, mode='constant', constant_values=0)
    prjr = np.pad(prjr, npad, mode='constant', constant_values=0)

    # save

    dxchange.write_tiff(prjl, proc + 'prepared/tif/0007_tomogram3-stxm_cl-135-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'prepared/tif/0007_tomogram3-stxm_cr-135-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'prepared/tif/0007_tomogram3-ang-135-cor', overwrite=True)
    np.save(proc + 'prepared/npy/0007_tomogram3-stxm_cl-135-cor.npy', prjl)
    np.save(proc + 'prepared/npy/0007_tomogram3-stxm_cr-135-cor.npy', prjr)
    np.save(proc + 'prepared/npy/0007_tomogram3-ang-135-cor.npy', ang)

    # 0008_tomogram4
    prjl = np.load(base + '0008_tomogram4-stxm_cl-178.npy') #17
    prjr = np.load(base + '0008_tomogram4-stxm_cr-178.npy') # 16 80
    ang = np.load(base + '0008_tomogram4-ang-178.npy')
    
    prjl = np.delete(prjl, [15, 16, 79], axis=0)
    prjr = np.delete(prjr, [15, 16, 79], axis=0)
    ang = np.delete(ang, [15, 16, 79], axis=0)

    # load 4

    dxchange.write_tiff(prjl, proc + 'raw/tif/0008_tomogram4-stxm_cl-175-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'raw/tif/0008_tomogram4-stxm_cr-175-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'raw/tif/0008_tomogram4-ang-175-cor', overwrite=True)
    np.save(proc + 'raw/npy/0008_tomogram4-stxm_cl-175-cor.npy', prjl)
    np.save(proc + 'raw/npy/0008_tomogram4-stxm_cr-175-cor.npy', prjr)
    np.save(proc + 'raw/npy/0008_tomogram4-ang-175-cor.npy', ang)

    #save 4
    prjl = prjl / scl
    prjr = prjr / scl
    prjl[prjl > 1] = 0.995
    prjr[prjr > 1] = 0.995
    prjl = tomopy.normalize_bg(prjl)
    prjl = tomopy.minus_log(prjl)
    prjr = tomopy.normalize_bg(prjr)
    prjr = tomopy.minus_log(prjr)
    npad = ((0, 0), (30, 0), (0, 0))
    prjl = np.pad(prjl, npad, mode='constant', constant_values=0)
    prjr = np.pad(prjr, npad, mode='constant', constant_values=0)
    dxchange.write_tiff(prjl, proc + 'prepared/tif/0008_tomogram4-stxm_cl-175-cor', overwrite=True)
    dxchange.write_tiff(prjr, proc + 'prepared/tif/0008_tomogram4-stxm_cr-175-cor', overwrite=True)
    dxchange.write_tiff(ang, proc + 'prepared/tif/0008_tomogram4-ang-175-cor', overwrite=True)
    np.save(proc + 'prepared/npy/0008_tomogram4-stxm_cl-175-cor.npy', prjl)
    np.save(proc + 'prepared/npy/0008_tomogram4-stxm_cr-175-cor.npy', prjr)
    np.save(proc + 'prepared/npy/0008_tomogram4-ang-175-cor.npy', ang)


'''
For each of the 2.1 micron and 5.4 micron NFB samples, the left circular (CL) and right circular (CR) projections 
are concatenated using the np.r_[arr1, arr2] function, alongwith the corresponding projection angles.

Subsequently, using Tomopy's joint reconstruction and reprojection alignment method 
https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.alignment.html#tomopy.prep.alignment.align_joint
is used to align the  together for 2.1 micron and 5.4 micron sample.
'''
def align_projs(base, proc):
    # 0002_tomogram1
    prjl = np.load(proc + 'prepared/npy/0002_tomogram1-stxm_cl-150-cor.npy')
    prjr = np.load(proc + 'prepared/npy/0002_tomogram1-stxm_cr-150-cor.npy')
    ang = np.load(proc + 'prepared/npy/0002_tomogram1-ang-150-cor.npy')
    print('ang 0002_tomogram1', ang)
    prj = np.r_[prjl, prjr]
    ang = np.r_[ang, ang]
    prj, sx, sy, conv = tomopy.align_joint(
            prj, ang, fdir=proc + 'aligned/0002_tomogram1/', iters=10, pad=(0, 0),
            blur=True, center=None, algorithm='sirt',
            upsample_factor=100, rin=0.5, rout=0.8,
            save=True, debug=True)
    dxchange.write_tiff(prj[0:150], proc + 'aligned/0002_tomogram1-stxm_cl-150-cor', overwrite=True)
    dxchange.write_tiff(prj[150:300], proc + 'aligned/0002_tomogram1-stxm_cr-150-cor', overwrite=True)
    np.save(proc + 'aligned/0002_tomogram1-sx.npy', sx[0:150])
    np.save(proc + 'aligned/0002_tomogram1-sy.npy', sy[150:300])
    
    # 0004_tomogram2
    prjl = np.load(proc + 'prepared/npy/0004_tomogram2-stxm_cl-50-cor.npy')
    prjr = np.load(proc + 'prepared/npy/0004_tomogram2-stxm_cr-50-cor.npy')
    ang = np.load(proc + 'prepared/npy/0004_tomogram2-ang-50-cor.npy')
    print('ang 0004_tomogram2', ang)
    prj = np.r_[prjl, prjr]
    ang = np.r_[ang, ang]
    prj, sx, sy, conv = tomopy.align_joint(
            prj, ang, fdir=proc + 'aligned/0004_tomogram2/', iters=10, pad=(0, 0),
            blur=True, center=None, algorithm='sirt',
            upsample_factor=100, rin=0.5, rout=0.8,
            save=True, debug=True)
    dxchange.write_tiff(prj[0:50], proc + 'aligned/0004_tomogram2-stxm_cl-50-cor', overwrite=True)
    dxchange.write_tiff(prj[50:100], proc + 'aligned/0004_tomogram2-stxm_cr-50-cor', overwrite=True)
    np.save(proc + 'aligned/0004_tomogram2-sx.npy', sx[0:50])
    np.save(proc + 'aligned/0004_tomogram2-sy.npy', sy[50:100])

    # 0007_tomogram3
    prjl = np.load(proc + 'prepared/npy/0007_tomogram3-stxm_cl-135-cor.npy')
    prjr = np.load(proc + 'prepared/npy/0007_tomogram3-stxm_cr-135-cor.npy')
    ang = np.load(proc + 'prepared/npy/0007_tomogram3-ang-135-cor.npy')
    print('ang 0007_tomogram3', ang)
    prj = np.r_[prjl, prjr]
    ang = np.r_[ang, ang]
    prj, sx, sy, conv = tomopy.align_joint(
            prj, ang, fdir=proc + 'aligned/0007_tomogram3/', iters=10, pad=(0, 0),
            blur=True, center=None, algorithm='sirt',
            upsample_factor=100, rin=0.5, rout=0.8,
            save=True, debug=True)
    dxchange.write_tiff(prj[0:135], proc + 'aligned/0007_tomogram3-stxm_cl-135-cor', overwrite=True)
    dxchange.write_tiff(prj[135:270], proc + 'aligned/0007_tomogram3-stxm_cr-135-cor', overwrite=True)
    np.save(proc + 'aligned/0007_tomogram3-sx.npy', sx[0:135])
    np.save(proc + 'aligned/0007_tomogram3-sy.npy', sy[135:270])
    
    # 0008_tomogram4
    prjl = np.load(proc + 'prepared/npy/0008_tomogram4-stxm_cl-175-cor.npy')
    prjr = np.load(proc + 'prepared/npy/0008_tomogram4-stxm_cr-175-cor.npy')
    ang = np.load(proc + 'prepared/npy/0008_tomogram4-ang-175-cor.npy')
    print('ang 0008_tomogram4', ang)
    prj = np.r_[prjl, prjr]
    ang = np.r_[ang, ang]
    prj, sx, sy, conv = tomopy.align_joint(
            prj, ang, fdir=proc + 'aligned/0008_tomogram4/', iters=10, pad=(0, 0),
            blur=True, center=None, algorithm='sirt',
            upsample_factor=100, rin=0.5, rout=0.8,
            save=True, debug=True)
    dxchange.write_tiff(prj[0:175], proc + 'aligned/0008_tomogram4-stxm_cl-175-cor', overwrite=True)
    dxchange.write_tiff(prj[175:350], proc + 'aligned/0008_tomogram4-stxm_cr-175-cor', overwrite=True)
    np.save(proc + 'aligned/0008_tomogram4-sx.npy', sx[0:175])
    np.save(proc + 'aligned/0008_tomogram4-sy.npy', sy[175:350])


'''
For each of the 2.1 micron and 5.4 micron NFB samples, the aligned left circular (CL) and right circular (CR) projections 
are used to do scalar 3D reconstruction of the samples. 
'''
def recon_aligned(base, proc):
    # These are scalar reconstructions for aligning the reconstructed object
    # 0002_tomogram1
    prjl = dxchange.read_tiff(proc + 'aligned/0002_tomogram1-stxm_cl-150-cor.tiff')
    prjr = dxchange.read_tiff(proc + 'aligned/0002_tomogram1-stxm_cr-150-cor.tiff')
    ang = np.load(proc + 'prepared/npy/0002_tomogram1-ang-150-cor.npy')
    recl = tomopy.recon(prjl, ang, algorithm='gridrec')
    recr = tomopy.recon(prjr, ang, algorithm='gridrec')
    dxchange.write_tiff(recl, proc + 'reconstructed/0002_tomogram1-stxm_cl-150-cor', overwrite=True)
    dxchange.write_tiff(recr, proc + 'reconstructed/0002_tomogram1-stxm_cr-150-cor', overwrite=True)

    # 0004_tomogram2
    prjl = dxchange.read_tiff(proc + 'aligned/0004_tomogram2-stxm_cl-50-cor.tiff')
    prjr = dxchange.read_tiff(proc + 'aligned/0004_tomogram2-stxm_cr-50-cor.tiff')
    ang = np.load(proc + 'prepared/npy/0004_tomogram2-ang-50-cor.npy')
    recl = tomopy.recon(prjl, ang, algorithm='gridrec')
    recr = tomopy.recon(prjr, ang, algorithm='gridrec')
    dxchange.write_tiff(recl, proc + 'reconstructed/0004_tomogram2-stxm_cl-50-cor', overwrite=True)
    dxchange.write_tiff(recr, proc + 'reconstructed/0004_tomogram2-stxm_cr-50-cor', overwrite=True)

    # 0007_tomogram3
    prjl = dxchange.read_tiff(proc + 'aligned/0007_tomogram3-stxm_cl-135-cor.tiff')
    prjr = dxchange.read_tiff(proc + 'aligned/0007_tomogram3-stxm_cr-135-cor.tiff')
    ang = np.load(proc + 'prepared/npy/0007_tomogram3-ang-135-cor.npy')
    recl = tomopy.recon(prjl, ang, algorithm='gridrec')
    recr = tomopy.recon(prjr, ang, algorithm='gridrec')
    dxchange.write_tiff(recl, proc + 'reconstructed/0007_tomogram3-stxm_cl-135-cor', overwrite=True)
    dxchange.write_tiff(recr, proc + 'reconstructed/0007_tomogram3-stxm_cr-135-cor', overwrite=True)
    
    # 0008_tomogram4
    prjl = dxchange.read_tiff(proc + 'aligned/0008_tomogram4-stxm_cl-175-cor.tiff')
    prjr = dxchange.read_tiff(proc + 'aligned/0008_tomogram4-stxm_cr-175-cor.tiff')
    ang = np.load(proc + 'prepared/npy/0008_tomogram4-ang-175-cor.npy')
    recl = tomopy.recon(prjl, ang, algorithm='gridrec')
    recr = tomopy.recon(prjr, ang, algorithm='gridrec')
    dxchange.write_tiff(recl, proc + 'reconstructed/0008_tomogram4-stxm_cl-175-cor', overwrite=True)
    dxchange.write_tiff(recr, proc + 'reconstructed/0008_tomogram4-stxm_cr-175-cor', overwrite=True)


'''
The forwardtransform() is used to transform the reconstructed 3D scalar object from tilt-1 orientation to tilt-2 orientation.
'''
def forwardtransform(arr, tform):
    for m in range(arr.shape[0]):
        arr[m, :, :] = transform.rotate(arr[m, :, :], tform[0], mode='edge')
    for m in range(arr.shape[1]):
        arr[:, m, :] = transform.rotate(arr[:, m, :], tform[1], mode='edge')
    for m in range(arr.shape[0]):
        arr[m, :, :] = transform.rotate(arr[m, :, :], tform[2], mode='edge')
    arr = ndimage.shift(arr, (tform[3], tform[4], tform[5]), mode='constant', cval=0)
    return arr


'''
The backwardtransform() is used to transform the reconstructed 3D scalar object from tilt-2 orientation 
to tilt-1 orientation. This is reverse of the operation of the forwardtransform() function.
'''
def backwardtransform(arr, tform):
    arr = ndimage.shift(arr, (-tform[3], -tform[4], -tform[5]), mode='constant', cval=0)
    for m in range(arr.shape[0]):
        arr[m, :, :] = transform.rotate(arr[m, :, :], -tform[2], mode='edge')
    for m in range(arr.shape[1]):
        arr[:, m, :] = transform.rotate(arr[:, m, :], -tform[1], mode='edge')
    for m in range(arr.shape[0]):
        arr[m, :, :] = transform.rotate(arr[m, :, :], -tform[0], mode='edge')
    return arr


'''
The costfunc() is used to evaluate the sum of squared error (SSE) between the pixels from the 3D scalar object in tilt-1
orientation and the 3D scalar object transformed from tilt-2 to the tilt-1 orientation. The costfunc() is called during 
the registration of the reconstructed 3D scalar object in tilt-1 and the tilt-2 object transformed to tilt-1 orientation
in the register_tilts() function.
'''
def costfunc(arr1, arr2, tform, fname='', debug=True):
    arr1 = forwardtransform(arr1.copy(), tform.copy())
    diff = arr1 - arr2
    cost = np.sum(np.power(diff, 2))
    if debug is True:
        dxchange.write_tiff(diff[diff.shape[0]//2, :, :], proc + fname + 'tmpx/test.tiff')
        dxchange.write_tiff(diff[:, diff.shape[1]//2, :], proc + fname + 'tmpy/test.tiff')
        dxchange.write_tiff(diff[:, :, diff.shape[2]//2], proc + fname + 'tmpz/test.tiff')
    return cost


'''
The register_tilts() function is used to align the 3D scalar reconstructed object in tilt-1 with the tilt-2 orientation. 
This alignment is done using a gradient descent algorithm. The alignment is done with respect to the 3 rotation angles 
around the x-axis, y-axis and z-axis, alongwith the translation in the x-axis, y-axis and z-axis. 
'''
def register_tilts(base, proc):
    # 0002_tomogram1 & 0004_tomogram2
    rec1 = dxchange.read_tiff(proc + 'reconstructed/0002_tomogram1-stxm_cl-150-cor.tiff')
    rec2 = dxchange.read_tiff(proc + 'reconstructed/0004_tomogram2-stxm_cl-50-cor.tiff')

    lbound = np.array([-32, -32, -2, -6, -10, -12], dtype='float32')  # v, >(z), >(y)
    ubound = np.array([-22, -28, 8, 14, 10, 8], dtype='float32')
    mpoint = 0.5 * (lbound + ubound)

    # Parameter init
    vl = lbound.copy()
    vu = ubound.copy()
    vm = mpoint.copy()

    a = 0
    for nn in range(10): # For each CG iteration
        for qq in range(6): # For each coordinate
            # Lower bound cost
            vl[qq] = lbound[qq]
            print (vl)
            costl = costfunc(rec1, rec2, vl, fname='tilt/tilt1/')
            a += 1
            # Upper bound cost
            vu[qq] = ubound[qq]
            costu = costfunc(rec1, rec2, vu, fname='tilt/tilt1/')
            a += 1
                
            for it in range(10): # For each binary search iteration
                
                # Middle point cost
                vm[qq] = (vu[qq] + vl[qq]) * 0.5
                costm = costfunc(rec1, rec2, vm, fname='tilt/tilt1/')
                a += 1
                # Update points and bounds
                costlm = costl - costm
                costum = costu - costm
                if costlm > costum:
                    vl[qq] = vm[qq]
                    costl = costm
                else:
                    vu[qq] = vm[qq]
                    costu = costm
                vm[qq] = (vu[qq] + vl[qq]) * 0.5

                # Save/print results
                print (str(nn) + '-' + str(qq) + '-' + str(it) + ': ' + str(vm))
    np.save(proc + 'tilt/tilt1.npy', vm)

    # The final registration parameters for the 2.1 micron sample are:
    # -31.155273 -30.107422 3.522461 4.0097656 -0.6738281 -1.8144531

    
    # 0007_tomogram3 & 0008_tomogram4
    rec1 = dxchange.read_tiff(proc + 'reconstructed/0007_tomogram3-stxm_cl-135-cor.tiff')
    rec2 = dxchange.read_tiff(proc + 'reconstructed/0008_tomogram4-stxm_cl-175-cor.tiff')

    lbound = np.array([32, -25, -5, -6, -12, -2], dtype='float32') # v, >(z), >(y)
    ubound = np.array([42, -35, 5, 14, 8, 18], dtype='float32')
    mpoint = 0.5 * (lbound + ubound)

    # Parameter init
    vl = lbound.copy()
    vu = ubound.copy()
    vm = mpoint.copy()

    a = 0
    for nn in range(10): # For each CG iteration
        for qq in range(6): # For each coordinate
            # Lower bound cost
            vl[qq] = lbound[qq]
            costl = costfunc(rec1, rec2, vl, fname='tilt/tilt2/')
            a += 1
            # Upper bound cost
            vu[qq] = ubound[qq]
            costu = costfunc(rec1, rec2, vu, fname='tilt/tilt2/')
            a += 1
                
            for it in range(10): # For each binary search iteration
                
                # Middle point cost
                vm[qq] = (vu[qq] + vl[qq]) * 0.5
                costm = costfunc(rec1, rec2, vm, fname='tilt/tilt2/')
                a += 1
                # Update points and bounds
                costlm = costl - costm
                costum = costu - costm
                if costlm > costum:
                    vl[qq] = vm[qq]
                    costl = costm
                else:
                    vu[qq] = vm[qq]
                    costu = costm
                vm[qq] = (vu[qq] + vl[qq]) * 0.5

                # Save/print results
                print (str(nn) + '-' + str(qq) + '-' + str(it) + ': ' + str(vm))
    np.save(proc + 'tilt/tilt2.npy', vm)

    # The final registration parameters for the 5.4 micron sample are:
    # 41.487305 -31.37207 0.8154297 4.908203 -1.4433594 6.9941406

'''
Once the 6 registration parameters are obtained through the gradient_descent() algorithm, 
these parameters are validated using the check_registration() function, where the tilt-1 reconstructed 
object is transformed to the tilt-2 orientation and the difference between them are computed in order
to find the mismatch.
'''
def check_registration(base, proc):
    rec1 = dxchange.read_tiff(proc + 'reconstructed/0002_tomogram1-stxm_cl-150-cor.tiff').copy()
    rec2 = dxchange.read_tiff(proc + 'reconstructed/0004_tomogram2-stxm_cr-50-cor.tiff').copy()
    tform = np.array([-31.155273, -30.107422, 3.522461, 4.0097656, -0.6738281, -1.8144531])
    rec1 = forwardtransform(rec1, tform)
    dxchange.write_tiff(rec1 - rec2, proc + 'validate/test1.tiff', overwrite=True)

    rec1 = dxchange.read_tiff(proc + 'reconstructed/0007_tomogram3-stxm_cl-135-cor.tiff').copy()
    rec2 = dxchange.read_tiff(proc + 'reconstructed/0008_tomogram4-stxm_cr-175-cor.tiff').copy()
    tform = np.array([41.487305, -31.37207, 0.8154297, 4.908203, -1.4433594, 6.9941406])
    rec1 = forwardtransform(rec1, tform)
    dxchange.write_tiff(rec1 - rec2, proc + 'validate/test2.tiff', overwrite=True)


'''
The recon_tilts() function is called in order to perform the Vector 3D Reconstruction of the magnetization domains 
from the CL and CR projections. The function reconstructs for both the 2.1 micron and 5.4 micron samples. 
'''
def recon_tilts(base, proc):

    # This is for the 5.4 micron NFB sample
    # Import the CL and CR projection angles
    # 0007_tomogram3 & 0008_tomogram4
    ang1 = np.load(proc + 'prepared/npy/0007_tomogram3-ang-135-cor.npy')
    ang2 = np.load(proc + 'prepared/npy/0008_tomogram4-ang-175-cor.npy')
   
    '''
    print('ang1', ang1*180/np.pi, ang1.shape)
    print('ang1[1, 36, 75, 122]', ang1[1]*180/np.pi, ang1[36]*180/np.pi, ang1[75]*180/np.pi, ang1[122]*180/np.pi)
    print('ang2', ang2*180/np.pi, ang2.shape)
    print('ang2[1, 35, 74, 120]', ang2[1]*180/np.pi, ang2[35]*180/np.pi, ang2[74]*180/np.pi, ang2[120]*180/np.pi)
    blabla
    '''

    # Import the CL and CR projections for the tilt-1 orientation
    # 0007_tomogram3
    prj1l = dxchange.read_tiff(proc + 'aligned/0007_tomogram3-stxm_cl-135-cor.tiff')
    prj1r = dxchange.read_tiff(proc + 'aligned/0007_tomogram3-stxm_cr-135-cor.tiff')
    dxchange.write_tiff(prj1l - prj1r, proc + 'final/recon2/prj1.tiff', overwrite=True)
    
    # Import the CL and CR projections for the tilt-2 orientation
    # 0008_tomogram4
    prj2l = dxchange.read_tiff(proc + 'aligned/0008_tomogram4-stxm_cl-175-cor.tiff')
    prj2r = dxchange.read_tiff(proc + 'aligned/0008_tomogram4-stxm_cr-175-cor.tiff')
    dxchange.write_tiff(prj2l - prj2r, proc + 'final/recon2/prj2.tiff', overwrite=True)

    print('sample2')

    # recon init
    # Initialization the 3D reconstructed values with zeros for tilt-1 orientation
    recx = np.zeros((111, 111, 111), dtype='float32')
    recy = np.zeros((111, 111, 111), dtype='float32')
    recz = np.zeros((111, 111, 111), dtype='float32')

    # Initialization the 3D reconstructed values with zeros for tilt-2 orientation
    recx4 = np.zeros((111, 111, 111), dtype='float32')
    recy4 = np.zeros((111, 111, 111), dtype='float32')
    recz4 = np.zeros((111, 111, 111), dtype='float32')

    # Iterative reconstruction algorithm
    # Run for 300 epochs
    for m in range(300):

        print('iter:', m)
        # CL - CR projections provide with the magnetic signal for tilt-1 orientation
        prj1diff = prj1l - prj1r
         
        # vector reconstruction using the tilt-1 orientation only 
        # 2 components of the vector are only reconstructed
        recy, recz = tomopy.vector(prj1diff, ang1, recy, recz, num_iter=1)
        
        '''
        dxchange.write_tiff(recx, proc + 'final/recon2/tmp1/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon2/tmp1/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon2/tmp1/recz.tiff')
        '''
        
        # transform the vector reconstruction from tilt-1 orientation to tilt-2 orientation geometrically
        # transform
        tform = np.array([41.487305, -31.37207, 0.8154297, 4.908203, -1.4433594, 6.9941406])
        recx = forwardtransform(recx, tform)
        recy = forwardtransform(recy, tform)
        recz = forwardtransform(recz, tform)

        # transform the 3D magnetization vectors from tilt-1 orientation to tilt-2 orientation
        # project XYX
        rotmat = np.zeros((3, 3), dtype='float32')
        a1 = tform[2] * np.pi / 180.
        a2 = tform[1] * np.pi / 180.
        a3 = tform[0] * np.pi / 180.
        rotmat[0, 0] = np.cos(a2)
        rotmat[0, 1] = np.sin(a2) * np.sin(a3)
        rotmat[0, 2] = np.cos(a3) * np.sin(a2)
        rotmat[1, 0] = np.sin(a1) * np.sin(a2)
        rotmat[1, 1] = np.cos(a1) * np.cos(a3) - np.cos(a2) * np.sin(a1) * np.sin(a3)
        rotmat[1, 2] = -np.cos(a1) * np.sin(a3) - np.cos(a2) * np.cos(a3) * np.sin(a1)
        rotmat[2, 0] = -np.cos(a1) * np.sin(a2)
        rotmat[2, 1] = np.cos(a3) * np.sin(a1) + np.cos(a1) * np.cos(a2) * np.sin(a3)
        rotmat[2, 2] = np.cos(a1) * np.cos(a2) * np.cos(a3) - np.sin(a1) * np.sin(a3)
        rec = np.dot(rotmat, [recx.flatten(), recy.flatten(), recz.flatten()])
        recx = np.reshape(rec[0], recx.shape)
        recy = np.reshape(rec[1], recy.shape)
        recz = np.reshape(rec[2], recz.shape)
        
        '''
        dxchange.write_tiff(recx, proc + 'final/recon2/tmp2/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon2/tmp2/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon2/tmp2/recz.tiff')
        '''

        # CL - CR projections provide with the magnetic signal for tilt-2 orientation
        prj2diff = prj2l - prj2r
         
        # vector reconstruction using the tilt-2 orientation only
        # 2 components of the vector are only reconstructed
        recy, recz = tomopy.vector(prj2diff, ang2, recy, recz, num_iter=1)
        '''
        dxchange.write_tiff(recx, proc + 'final/recon2/tmp3/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon2/tmp3/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon2/tmp3/recz.tiff')
        '''

        # transform the vector reconstruction from tilt-2 orientation back to tilt-1 orientation geometrically
        # back transform
        recx = backwardtransform(recx, tform)
        recy = backwardtransform(recy, tform)
        recz = backwardtransform(recz, tform)

        # transform the 3D magnetization vectors from tilt-2 orientation back to tilt-1 orientation
        # project XYX
        rotmat = np.zeros((3, 3), dtype='float32')
        a1 = -tform[0] * np.pi / 180.
        a2 = -tform[1] * np.pi / 180.
        a3 = -tform[2] * np.pi / 180.
        rotmat[0, 0] = np.cos(a2)
        rotmat[0, 1] = np.sin(a2) * np.sin(a3)
        rotmat[0, 2] = np.cos(a3) * np.sin(a2)
        rotmat[1, 0] = np.sin(a1) * np.sin(a2)
        rotmat[1, 1] = np.cos(a1) * np.cos(a3) - np.cos(a2) * np.sin(a1) * np.sin(a3)
        rotmat[1, 2] = -np.cos(a1) * np.sin(a3) - np.cos(a2) * np.cos(a3) * np.sin(a1)
        rotmat[2, 0] = -np.cos(a1) * np.sin(a2)
        rotmat[2, 1] = np.cos(a3) * np.sin(a1) + np.cos(a1) * np.cos(a2) * np.sin(a3)
        rotmat[2, 2] = np.cos(a1) * np.cos(a2) * np.cos(a3) - np.sin(a1) * np.sin(a3)
        rec = np.dot(rotmat, [recx.flatten(), recy.flatten(), recz.flatten()])
        recx = np.reshape(rec[0], recx.shape)
        recy = np.reshape(rec[1], recy.shape)
        recz = np.reshape(rec[2], recz.shape)
        
        '''
        The error for the 3 vector components are computed for successive iterations in tilt-1 orientation.
        This is used for finding how the algorithm converges over the epochs.
        '''
        errx4 = np.square(np.subtract(recx, recx4)).sum(); erry4 = np.square(np.subtract(recy, recy4)).sum(); errz4 = np.square(np.subtract(recz, recz4)).sum()
        print('smaple2 mse recx, recy, recz', errx4, erry4, errz4)
        
        if m == 0:
            errx4a = np.array([errx4]); erry4a = np.array([erry4]); errz4a = np.array([errz4])
        else:
            errx4a = np.append(errx4a, errx4); erry4a = np.append(erry4a, erry4); errz4a = np.append(errz4a, errz4)
            
        recx4 = np.copy(recx); recy4 = np.copy(recy); recz4 = np.copy(recz)
        
        '''
        Save the reconstructions in each epoch
        '''
        dxchange.write_tiff(recx, proc + 'final/recon2/tmp4/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon2/tmp4/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon2/tmp4/recz.tiff')
   
    # Save the error files at the end of the epochs
    np.savetxt(proc + 'final/recon2/err4xa_sample2.txt', errx4a)
    np.savetxt(proc + 'final/recon2/err4ya_sample2.txt', erry4a)
    np.savetxt(proc + 'final/recon2/err4za_sample2.txt', errz4a)
    ##############

    # This is for the 2.1 micron NFB sample
    # Import the CL and CR projection angles
    # 0002_tomogram1 & 0004_tomogram2
    ang1 = np.load(proc + 'prepared/npy/0002_tomogram1-ang-150-cor.npy')
    ang2 = np.load(proc + 'prepared/npy/0004_tomogram2-ang-50-cor.npy')

    # Import the CL and CR projections for the tilt-1 orientation
    # 0002_tomogram1
    prj1l = dxchange.read_tiff(proc + 'aligned/0002_tomogram1-stxm_cl-150-cor.tiff')
    prj1r = dxchange.read_tiff(proc + 'aligned/0002_tomogram1-stxm_cr-150-cor.tiff')
    dxchange.write_tiff(prj1l - prj1r, proc + 'final/recon1/prj1.tiff', overwrite=True)

    # Import the CL and CR projections for the tilt-2 orientation
    # 0004_tomogram2
    prj2l = dxchange.read_tiff(proc + 'aligned/0004_tomogram2-stxm_cl-50-cor.tiff')
    prj2r = dxchange.read_tiff(proc + 'aligned/0004_tomogram2-stxm_cr-50-cor.tiff')
    dxchange.write_tiff(prj2l - prj2r, proc + 'final/recon1/prj2.tiff', overwrite=True)

    # recon init
    print('sample1')
    # Initialization the 3D reconstructed values with zeros for tilt-1 orientation
    recx = np.zeros((161, 161, 161), dtype='float32')
    recy = np.zeros((161, 161, 161), dtype='float32')
    recz = np.zeros((161, 161, 161), dtype='float32')
    
    # Initialization the 3D reconstructed values with zeros for tilt-2 orientation
    recx4 = np.zeros((161, 161, 161), dtype='float32')
    recy4 = np.zeros((161, 161, 161), dtype='float32')
    recz4 = np.zeros((161, 161, 161), dtype='float32')

    # Iterative reconstruction algorithm
    # Run for 100 epochs (faster convergence for the smaller NFB sample)
    for m in range(100):
        print('iter:', m)
        # CL - CR projections provide with the magnetic signal for tilt-1 orientation    
        prj1diff = prj1l - prj1r
        
        # vector reconstruction using the tilt-1 orientation only
        # 2 components of the vector are only reconstructed
        recy, recz = tomopy.vector(prj1diff, ang1, recy, recz, num_iter=1)
        
        '''
        dxchange.write_tiff(recx, proc + 'final/recon1/tmp1/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon1/tmp1/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon1/tmp1/recz.tiff')
        '''

        # transform the vector reconstruction from tilt-1 orientation to tilt-2 orientation geometrically
        # transform
        tform = np.array([-31.155273, -30.107422, 3.522461, 4.0097656, -0.6738281, -1.8144531])
        recx = forwardtransform(recx, tform)
        recy = forwardtransform(recy, tform)
        recz = forwardtransform(recz, tform)

        # transform the 3D magnetization vectors from tilt-1 orientation to tilt-2 orientation
        # project XYX
        rotmat = np.zeros((3, 3), dtype='float32')
        a1 = tform[2] * np.pi / 180.
        a2 = tform[1] * np.pi / 180.
        a3 = tform[0] * np.pi / 180.
        rotmat[0, 0] = np.cos(a2)
        rotmat[0, 1] = np.sin(a2) * np.sin(a3)
        rotmat[0, 2] = np.cos(a3) * np.sin(a2)
        rotmat[1, 0] = np.sin(a1) * np.sin(a2)
        rotmat[1, 1] = np.cos(a1) * np.cos(a3) - np.cos(a2) * np.sin(a1) * np.sin(a3)
        rotmat[1, 2] = -np.cos(a1) * np.sin(a3) - np.cos(a2) * np.cos(a3) * np.sin(a1)
        rotmat[2, 0] = -np.cos(a1) * np.sin(a2)
        rotmat[2, 1] = np.cos(a3) * np.sin(a1) + np.cos(a1) * np.cos(a2) * np.sin(a3)
        rotmat[2, 2] = np.cos(a1) * np.cos(a2) * np.cos(a3) - np.sin(a1) * np.sin(a3)
        rec = np.dot(rotmat, [recx.flatten(), recy.flatten(), recz.flatten()])
        recx = np.reshape(rec[0], recx.shape)
        recy = np.reshape(rec[1], recy.shape)
        recz = np.reshape(rec[2], recz.shape)
        '''
        dxchange.write_tiff(recx, proc + 'final/recon1/tmp2/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon1/tmp2/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon1/tmp2/recz.tiff')
        '''
        

        # CL - CR projections provide the magnetic signal for the tilt-2 orientation
        prj2diff = prj2l - prj2r
        
        # Vector reconstruction using the tilt-2 orientation only
        # 2 components of the vector are only reconstructed
        recy, recz = tomopy.vector(prj2diff, ang2, recy, recz, num_iter=1)
        '''
        dxchange.write_tiff(recx, proc + 'final/recon1/tmp3/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon1/tmp3/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon1/tmp3/recz.tiff')
        '''
        
        # transform the vector reconstruction from tilt-2 orientation back to tilt-1 orientation geometrically
        # back transform
        recx = backwardtransform(recx, tform)
        recy = backwardtransform(recy, tform)
        recz = backwardtransform(recz, tform)

        # transform the 3D magnetization vectors from tilt-2 orientation back to tilt-1 orientation
        # project XYX in backward direction
        rotmat = np.zeros((3, 3), dtype='float32')
        a1 = -tform[0] * np.pi / 180.
        a2 = -tform[1] * np.pi / 180.
        a3 = -tform[2] * np.pi / 180.
        rotmat[0, 0] = np.cos(a2)
        rotmat[0, 1] = np.sin(a2) * np.sin(a3)
        rotmat[0, 2] = np.cos(a3) * np.sin(a2)
        rotmat[1, 0] = np.sin(a1) * np.sin(a2)
        rotmat[1, 1] = np.cos(a1) * np.cos(a3) - np.cos(a2) * np.sin(a1) * np.sin(a3)
        rotmat[1, 2] = -np.cos(a1) * np.sin(a3) - np.cos(a2) * np.cos(a3) * np.sin(a1)
        rotmat[2, 0] = -np.cos(a1) * np.sin(a2)
        rotmat[2, 1] = np.cos(a3) * np.sin(a1) + np.cos(a1) * np.cos(a2) * np.sin(a3)
        rotmat[2, 2] = np.cos(a1) * np.cos(a2) * np.cos(a3) - np.sin(a1) * np.sin(a3)
        rec = np.dot(rotmat, [recx.flatten(), recy.flatten(), recz.flatten()])
        recx = np.reshape(rec[0], recx.shape)
        recy = np.reshape(rec[1], recy.shape)
        recz = np.reshape(rec[2], recz.shape)

        '''
        The error for the 3 vector components are computed for successive iterations in tilt-1 orientation.
        This is used for finding how the algorithm converges over the epochs.
        '''
        errx4 = np.square(np.subtract(recx, recx4)).sum(); erry4 = np.square(np.subtract(recy, recy4)).sum(); errz4 = np.square(np.subtract(recz, recz4)).sum()
        print('smaple1 mse recx, recy, recz', errx4, erry4, errz4)
         
        if m == 0:
            errx4a = np.array([errx4]); erry4a = np.array([erry4]); errz4a = np.array([errz4])
        else:
            errx4a = np.append(errx4a, errx4); erry4a = np.append(erry4a, erry4); errz4a = np.append(errz4a, errz4)
        
        recx4 = np.copy(recx); recy4 = np.copy(recy); recz4 = np.copy(recz)

        '''
        Save the reconstructions in each epoch in the tilt-1 orientation
        '''
        dxchange.write_tiff(recx, proc + 'final/recon1/tmp4/recx.tiff')
        dxchange.write_tiff(recy, proc + 'final/recon1/tmp4/recy.tiff')
        dxchange.write_tiff(recz, proc + 'final/recon1/tmp4/recz.tiff')
        ###############
  
    '''
    Save the text files with the errors in the magnetic domain reconstruction along the x-axis, y-axis and z-axis.
    '''
    np.savetxt(proc + 'final/recon1/err4xa_sample1.txt', errx4a)
    np.savetxt(proc + 'final/recon1/err4ya_sample1.txt', erry4a)
    np.savetxt(proc + 'final/recon1/err4za_sample1.txt', errz4a)
    ##############


'''
This is an auxilary function which aids in visualization of the reconstructed magnetization vectors.
'''
def recon_plot(base, proc):
    # recon1 is the 2.1 micron samples
    recx = dxchange.read_tiff(proc + 'final/recon1/tmp4/recx-9.tiff')
    recy = dxchange.read_tiff(proc + 'final/recon1/tmp4/recy-9.tiff')
    recz = dxchange.read_tiff(proc + 'final/recon1/tmp4/recz-9.tiff')
    plt.figure(figsize=[20, 6])
    dx = 3
    for m in range(10):
        plt.subplot(3, 10, m+1)
        plt.title('yz')
        u = recy[15 + m * 14, ::dx, ::dx]
        v = recz[15 + m * 14, ::dx, ::dx]
        plt.imshow(u, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)

    for m in range(10):
        plt.subplot(3, 10, 10 + m+1)
        plt.title('xz')
        u = recx[::dx, 45 + m * 7, ::dx]
        v = recz[::dx, 45 + m * 7, ::dx]
        plt.imshow(u, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)

    for m in range(10):
        plt.subplot(3, 10, 20 + m+1)
        plt.title('xy')
        u = recx[::dx, ::dx, 45 + m * 7]
        v = recy[::dx, ::dx, 45 + m * 7]
        plt.imshow(u, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)
    plt.tight_layout()
    plt.savefig(proc + 'final/quiver1-zxz.png', dpi=640)

    ####
    # recon2 is the 5.4 micron samples
    recx = dxchange.read_tiff(proc + 'final/recon2/tmp4/recx-9.tiff')
    recy = dxchange.read_tiff(proc + 'final/recon2/tmp4/recy-9.tiff')
    recz = dxchange.read_tiff(proc + 'final/recon2/tmp4/recz-9.tiff')
    plt.figure(figsize=[20, 6])
    dx = 3
    for m in range(10):
        plt.subplot(3, 10, m+1)
        plt.title('yz')
        u = recy[15 + m * 10, ::dx, ::dx]
        v = recz[15 + m * 10, ::dx, ::dx]
        plt.imshow(v, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)

    for m in range(10):
        plt.subplot(3, 10, 10 + m+1)
        plt.title('xz')
        u = recx[::dx, 10 + m * 9, ::dx]
        v = recz[::dx, 10 + m * 9, ::dx]
        plt.imshow(v, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)

    for m in range(10):
        plt.subplot(3, 10, 20 + m+1)
        plt.title('xy')
        u = recx[::dx, ::dx, 10 + m * 9]
        v = recy[::dx, ::dx, 10 + m * 9]
        plt.imshow(v, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)
    plt.tight_layout()
    plt.savefig(proc + 'final/quiver2-zxz.png', dpi=640)


'''
This is an auxilary function which rotates the magnetic vectors by certain angle of rotation.
'''
def test(base, proc):
    
    # recon init
    recx = np.zeros((100, 100, 100), dtype='float32')
    recy = np.zeros((100, 100, 100), dtype='float32')
    recz = np.zeros((100, 100, 100), dtype='float32')

    for m in range(20):
        recy[40 + m, 30:70, 20:80] = 1

    dxchange.write_tiff(recx, proc + 'test/recx.tiff', overwrite=True)
    dxchange.write_tiff(recy, proc + 'test/recy.tiff', overwrite=True)
    dxchange.write_tiff(recz, proc + 'test/recz.tiff', overwrite=True)

    # transform
    tform = np.array([0, 20, 0, 0, 0, 0])
    recx = forwardtransform(recx, tform)
    recy = forwardtransform(recy, tform)
    recz = forwardtransform(recz, tform)

    # project XYX
    rotmat = np.zeros((3, 3), dtype='float32')
    a1 = tform[0] * np.pi / 180.
    a2 = tform[1] * np.pi / 180.
    a3 = tform[2] * np.pi / 180.
    rotmat[0, 0] = np.cos(a2)
    rotmat[0, 1] = np.sin(a2) * np.sin(a3)
    rotmat[0, 2] = np.cos(a3) * np.sin(a2)
    rotmat[1, 0] = np.sin(a1) * np.sin(a2)
    rotmat[1, 1] = np.cos(a1) * np.cos(a3) - np.cos(a2) * np.sin(a1) * np.sin(a3)
    rotmat[1, 2] = -np.cos(a1) * np.sin(a3) - np.cos(a2) * np.cos(a3) * np.sin(a1)
    rotmat[2, 0] = -np.cos(a1) * np.sin(a2)
    rotmat[2, 1] = np.cos(a3) * np.sin(a1) - np.cos(a1) * np.cos(a2) * np.sin(a3)
    rotmat[2, 2] = np.cos(a1) * np.cos(a2) * np.cos(a3) - np.sin(a1) * np.sin(a3)
    rec = np.dot(rotmat, [recx.flatten(), recy.flatten(), recz.flatten()])
    recx = np.reshape(rec[0], recx.shape)
    recy = np.reshape(rec[1], recy.shape)
    recz = np.reshape(rec[2], recz.shape)

    dxchange.write_tiff(recx, proc + 'test/recx-tr.tiff', overwrite=True)
    dxchange.write_tiff(recy, proc + 'test/recy-tr.tiff', overwrite=True)
    dxchange.write_tiff(recz, proc + 'test/recz-tr.tiff', overwrite=True)

    # recx = backwardtransform(recx, tform)
    # recy = backwardtransform(recy, tform)
    # recz = backwardtransform(recz, tform)

    # dxchange.write_tiff(recx, proc + 'test/recx-bw.tiff', overwrite=True)
    # dxchange.write_tiff(recy, proc + 'test/recy-bw.tiff', overwrite=True)
    # dxchange.write_tiff(recz, proc + 'test/recz-bw.tiff', overwrite=True)

    plt.figure(figsize=[20, 6])
    dx = 3
    for m in range(10):
        plt.subplot(3, 10, m+1)
        plt.title('yz')
        u = recy[5 + m * 10, ::dx, ::dx]
        v = recz[5 + m * 10, ::dx, ::dx]
        plt.imshow(v, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)

    for m in range(10):
        plt.subplot(3, 10, 10 + m+1)
        plt.title('xz')
        u = recx[::dx, 5 + m * 10, ::dx]
        v = recz[::dx, 5 + m * 10, ::dx]
        plt.imshow(v, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)

    for m in range(10):
        plt.subplot(3, 10, 20 + m+1)
        plt.title('xy')
        u = recx[::dx, ::dx, 5 + m * 10]
        v = recy[::dx, ::dx, 5 + m * 10]
        plt.imshow(v, vmax=0.00015, vmin=-0.00015)
        plt.quiver(u, v)
    plt.tight_layout()
    plt.savefig(proc + 'test/quiver.png', dpi=640)


if __name__ == "__main__":
    base = 'Data_NanoMAX/staged/'
    proc = 'Data_NanoMAX/processed/'
    
    # The following functions are called sequentially
    prep_projs(base, proc)
    align_projs(base, proc)
    recon_aligned(base, proc)
    register_tilts(base, proc)
    check_registration(base, proc)
    recon_tilts(base, proc)

    # recon_plot() and test() are auxiliary functions for visualization as well 
    # as testing the rotation of magnetic vectors by desired rotation angle 
    recon_plot(base, proc)
    test(base, proc)