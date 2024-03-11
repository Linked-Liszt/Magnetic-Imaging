# Magnetic-Imaging
Magnetization Vector Reconstruction Algorithm based on STXM - Tomography Experiments

STXM - Tomography Experiments have been done in the NanoMAX beamline at the MAX-IV Synchrotron in Lund, Sweden on Nd<sub>2</sub>Fe<sub>14</sub>B permanent magnet samples.

The data files are available upon request.

The reconstruction steps and the algorithm has been implemented in Tomopy. All the code (in stepwise manner) is in Magnetization_Vector_Recon.py.

The 3D magnetization domains are then rotated in the direction of maximum contrast. For the 5.4 micron NFB sample, it is in rotate_3Dobj.py.

Finally, the scalar reconstruction of the object is done also in Tomopy, using only one tilt orientation. Subsequently, this 3D scalar magnetization is rotated appropriately around the y-axis and x-axis successively. Based on this rotated scalar reconstruction, the 3D vector magnetization of the object only is isolated. This is in rotate_scal_3Dobj_thresh.py for the 5.4 micron NFB sample.

Using these magnetization vectors in 3D, the vector field visualization is done in Paraview.
