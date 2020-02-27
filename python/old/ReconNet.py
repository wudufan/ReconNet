
# coding: utf-8

# In[1]:


from ctypes import *
import os
import numpy as np
import re
import tensorflow as tf


# In[2]:


if __name__ == '__main__':
    module = tf.load_op_library('./libReconNet4D.so')
    cmodule = cdll.LoadLibrary('./libReconNet4D.so')
else:
    module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'libReconNet4D.so'))
    cmodule = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libReconNet4D.so'))


# In[3]:


class ReconNet:    
    def __init__(self):
        self.rotview = 720
        self.nu = 736
        self.nv = 1
        self.nx = 512
        self.ny = 512
        self.nz = 1
        self.dx = 1
        self.dy = 1
        self.dz = 2
        self.cx = 0
        self.cy = 0
        self.cz = 0
        self.dsd = 1085.6
        self.dso = 595
        self.du = 1.2858
        self.dv = 2
        self.da = self.du/self.dsd
        self.off_a = 0
        self.off_u = 0
        self.off_v = 0

    def FromFile(self, filename):
        f = open(filename, 'r')
        params = dict()
        for line in f:
            substrs = re.split(' |\t|\n', line)
            name = substrs[0].lower()
            var = np.float32(substrs[1])
            params[name] = var
        for varname in dir(self):
            if varname in params:
                setattr(self, varname, params[varname])
        self.rotview = int(self.rotview)
        self.nu = int(self.nu)
        self.nv = int(self.nv)
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.nz = int(self.nz)
        self.off_a = -self.off_a
        self.da = self.du / self.dsd
        self.UpdateAngularSampling()
    
    def UpdateAngularSampling(self):
        self.angles = (np.arange(0, self.rotview, dtype=np.float32) * 2 * np.pi / self.rotview).tolist()

    def get_det_sz(self, type_projector = 0):
        return [self.da, self.dv]
        
    def get_det_off(self, type_projector = 0):
        return [self.off_a, self.off_v]
    
    def cSetDevice(self, device):
        return cmodule.SetDevice(c_int(device))
    
    def backprojection4d(self, sino, angles, voxel_sz, img_shape=None, type_projector=0, name=None):
        if img_shape is None:
            img_shape = [self.nx, self.ny, self.nz]
        
        return module.backprojection4d(sino = sino, 
                                       angles = angles,
                                       voxel_sz = voxel_sz,
                                       output_shape = img_shape, 
                                       det_sz = self.get_det_sz(type_projector), 
                                       det_off = self.get_det_off(type_projector), 
                                       dsd = self.dsd,
                                       dso = self.dso, 
                                       type_projector = type_projector, 
                                       name = name)

    def backprojection3d(self, sino, angles, voxel_sz, img_shape=None, type_projector=0, name=None):
        img = self.backprojection4d(tf.expand_dims(sino, -2), angles, voxel_sz, img_shape, type_projector, name)
        return tf.squeeze(img, -2)
    
    def projection4d(self, img, angles, voxel_sz, sino_shape=None, type_projector=0, name=None):
        if sino_shape is None:
            sino_shape = [self.nu, self.rotview, self.nv]
        
        return module.projection4d(img = img,
                                   angles = angles,
                                   voxel_sz = voxel_sz,
                                   output_shape = sino_shape, 
                                   det_sz = self.get_det_sz(type_projector), 
                                   det_off = self.get_det_off(type_projector), 
                                   dsd = self.dsd,
                                   dso = self.dso, 
                                   name = name)
    
    def projection3d(self, img, angles, voxel_sz, sino_shape=None, type_projector=0, name=None):
        sino = self.projection4d(tf.expand_dims(img, -2), angles, voxel_sz, sino_shape, type_projector, name)
        return tf.squeeze(sino, -2)
    
    def cFilter4d(self, sino, type_filter = 0, type_projector = 0):
        sino = sino.astype(np.float32)
        fsino = np.zeros(sino.shape, np.float32)
        
        cmodule.cFilterFanFilter(fsino.ctypes.data_as(POINTER(c_float)), 
                                 sino.ctypes.data_as(POINTER(c_float)),
                                 c_int(sino.shape[0]), c_int(sino.shape[4]), 
                                 c_int(sino.shape[1]), c_int(sino.shape[2]), c_int(sino.shape[3]), 
                                 c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v), 
                                 c_float(self.dsd), c_float(self.dso), c_int(type_filter), c_int(type_projector))
        
        return fsino
    
    def cFilter3d(self, sino, type_filter = 0, type_projector = 0):
        fsino = self.cFilter4d(sino[..., np.newaxis, :], type_filter, type_projector)
        return fsino[..., 0, :]
    
    def cPixelBackprojection4d(self, sino, angles = None, type_projector = 0):
        sino = sino.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        img = np.zeros([sino.shape[0], self.nx, self.ny, self.nz, sino.shape[4]], np.float32)
        
        cmodule.cPixelDrivenFanBackprojection(
            img.ctypes.data_as(POINTER(c_float)), 
            sino.ctypes.data_as(POINTER(c_float)), 
            angles.ctypes.data_as(POINTER(c_float)), 
            c_int(img.shape[0]), c_int(img.shape[4]), c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]),
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_int(sino.shape[1]), c_int(sino.shape[2]), c_int(sino.shape[3]),
            c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v),
            c_float(self.dsd), c_float(self.dso),
            c_int(type_projector))
        
        return img
    
    def cPixelBackprojection3d(self, sino, angles = None, type_projector = 0):
        img = self.cPixelBackprojection4d(sino[..., np.newaxis, :], angles, type_projector)        
        return img[..., 0, :]
    
    def cSiddonFanProjection4d(self, img, angles = None, type_projector = 0):
        img = img.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        sino = np.zeros([img.shape[0], self.nu, self.rotview, self.nv, img.shape[4]], np.float32)
        
        cmodule.cSiddonFanProjection(
            sino.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            angles.ctypes.data_as(POINTER(c_float)), 
            c_int(img.shape[0]), c_int(img.shape[4]), c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]),
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_int(sino.shape[1]), c_int(sino.shape[2]), c_int(sino.shape[3]),
            c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v),
            c_float(self.dsd), c_float(self.dso),
            c_int(type_projector))
        
        return sino
    
    def cSiddonFanProjection3d(self, img, angles = None, type_projector = 0):
        sino = self.cSiddonFanProjection4d(img[..., np.newaxis, :], angles, type_projector)
        return sino[..., 0, :]
    
    def cSiddonFanBackprojection4d(self, sino, angles = None, type_projector = 0):
        sino = sino.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        img = np.zeros([sino.shape[0], self.nx, self.ny, self.nz, sino.shape[4]], np.float32)
        
        cmodule.cSiddonFanBackprojection(
            img.ctypes.data_as(POINTER(c_float)), 
            sino.ctypes.data_as(POINTER(c_float)), 
            angles.ctypes.data_as(POINTER(c_float)), 
            c_int(img.shape[0]), c_int(img.shape[4]), c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]),
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_int(sino.shape[1]), c_int(sino.shape[2]), c_int(sino.shape[3]),
            c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v),
            c_float(self.dsd), c_float(self.dso),
            c_int(type_projector))
        
        return img
    
    def cSiddonFanBackprojection3d(self, sino, angles = None, type_projector = 0):
        img = self.cSiddonFanBackprojection4d(sino[..., np.newaxis, :], angles, type_projector)        
        return img[..., 0, :]
    
    def cSiddonFanNormImg4d(self, sino, wsino = None, angles = None, type_projector = 0):
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
            
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
            
        fp = self.cSiddonFanProjection4d(np.ones([sino.shape[0], self.nx, self.ny, self.nz, sino.shape[4]], np.float32), 
                                         angles, type_projector)
        bp = self.cSiddonFanBackprojection4d(fp * wsino, angles, type_projector)
        return bp
    
    def cSiddonFanNormImg3d(self, sino, wsino = None, angles = None, type_projector = 0):
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
            
        normImg = self.cSiddonFanNormImg4d(sino[..., np.newaxis, :], wsino[..., np.newaxis, :], angles, type_projector)
        return normImg[..., 0, :]
    
    def cSiddonFanOSSQS4d(self, nSubsets, img, sino, normImg, wsino = None, angles = None, order = None, type_projector = 0, lam = 1):
        img = img.astype(np.float32)
        sino = sino.astype(np.float32)
        normImg = normImg.astype(np.float32)
        
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
            
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        res = np.zeros(img.shape, np.float32)
        nviewPerSubset = int(np.ceil(sino.shape[2] / float(nSubsets)))
        
        if order is not None:
            angles = angles[order]
            sino = np.copy(sino[..., order, :, :], 'C')
            wsino = np.copy(wsino[..., order, :, :], 'C')
        
        cmodule.cOSSQS(
            res.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            sino.ctypes.data_as(POINTER(c_float)), 
            normImg.ctypes.data_as(POINTER(c_float)), 
            wsino.ctypes.data_as(POINTER(c_float)), 
            angles.ctypes.data_as(POINTER(c_float)), 
            c_int(nviewPerSubset), c_int(nSubsets),
            c_int(img.shape[0]), c_int(img.shape[4]), c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]),
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_int(sino.shape[1]), c_int(sino.shape[2]), c_int(sino.shape[3]),
            c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v),
            c_float(self.dsd), c_float(self.dso),
            c_int(type_projector), c_float(lam))
        
        return res
    
    def cSiddonFanOSSQS3d(self, nSubsets, img, sino, normImg, wsino = None, angles = None, order = None, type_projector = 0, lam = 1):
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
        
        res = self.cSiddonFanOSSQS4d(nSubsets, img[..., np.newaxis, :], sino[..., np.newaxis, :], 
                                     normImg[..., np.newaxis, :], wsino[..., np.newaxis, :], 
                                     angles, order, type_projector, lam)
        
        return res[..., 0, :]
    
    def cSiddonParallelProjection4d(self, img, detCenter, detU, detV, invRayDir):
        img = img.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = detU.astype(np.float32)
        detV = detV.astype(np.float32)
        invRayDir = invRayDir.astype(np.float32)
        
        prj = np.zeros([img.shape[0], self.nu, self.rotview, self.nv, img.shape[-1]], np.float32)
        cmodule.cSiddonParallelProjection(
            prj.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            detCenter.ctypes.data_as(POINTER(c_float)), 
            detU.ctypes.data_as(POINTER(c_float)), 
            detV.ctypes.data_as(POINTER(c_float)), 
            invRayDir.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), c_int(img.shape[-1]), 
            c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(self.nu), c_int(self.rotview), c_int(self.nv),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return prj
    
    def cSiddonParallelBackprojection4d(self, prj, detCenter, detU, detV, invRayDir):
        prj = prj.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = detU.astype(np.float32)
        detV = detV.astype(np.float32)
        invRayDir = invRayDir.astype(np.float32)
        
        img = np.zeros([prj.shape[0], self.nx, self.ny, self.nz, prj.shape[-1]], np.float32)
        cmodule.cSiddonParallelBackprojection(
            img.ctypes.data_as(POINTER(c_float)), 
            prj.ctypes.data_as(POINTER(c_float)), 
            detCenter.ctypes.data_as(POINTER(c_float)), 
            detU.ctypes.data_as(POINTER(c_float)), 
            detV.ctypes.data_as(POINTER(c_float)), 
            invRayDir.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), c_int(img.shape[-1]), 
            c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(self.nu), c_int(self.rotview), c_int(self.nv),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return img
    
    def cSiddonConeProjectionAbitrary4d(self, img, detCenter, detU, detV, src):
        img = img.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = detU.astype(np.float32)
        detV = detV.astype(np.float32)
        src = src.astype(np.float32)
        
        prj = np.zeros([img.shape[0], self.nu, self.rotview, self.nv, img.shape[-1]], np.float32)
        cmodule.cSiddonConeProjectionAbitrary(
            prj.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            detCenter.ctypes.data_as(POINTER(c_float)), 
            detU.ctypes.data_as(POINTER(c_float)), 
            detV.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), c_int(img.shape[-1]), 
            c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(self.nu), c_int(self.rotview), c_int(self.nv),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return prj
    
    def cSiddonConeBackprojectionAbitrary4d(self, prj, detCenter, detU, detV, src):
        prj = prj.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = detU.astype(np.float32)
        detV = detV.astype(np.float32)
        src = src.astype(np.float32)
        
        img = np.zeros([prj.shape[0], self.nx, self.ny, self.nz, prj.shape[-1]], np.float32)
        cmodule.cSiddonConeBackprojectionAbitrary(
            img.ctypes.data_as(POINTER(c_float)), 
            prj.ctypes.data_as(POINTER(c_float)), 
            detCenter.ctypes.data_as(POINTER(c_float)), 
            detU.ctypes.data_as(POINTER(c_float)), 
            detV.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), c_int(img.shape[-1]), 
            c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(self.nu), c_int(self.rotview), c_int(self.nv),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return img
    
    def get_img_shape(self):
        return [self.nx, self.ny, self.nz]
    
    def get_sino_shape(self):
        return [self.nu, self.rotview, self.nv]
    


# In[5]:


from tensorflow.python.framework import ops

@ops.RegisterGradient("Backprojection4D")
def _backprojection4d_grad(op, grad):
    g = module.projection4d(img = grad, 
                            angles = op.inputs[1],
                            voxel_sz = op.inputs[2],
                            output_shape = op.inputs[0].shape[1:4].as_list(),
                            det_sz = op.get_attr('det_sz'), 
                            det_off = op.get_attr('det_off'),
                            dsd = op.get_attr('dsd'),
                            dso = op.get_attr('dso'),
                            type_projector = op.get_attr('type_projector'))
    
    return g, None, None
    

@ops.RegisterGradient("Projection4D")
def _projection4d_grad(op, grad):
    g = module.backprojection4d(sino = grad,
                                angles = op.inputs[1],
                                voxel_sz = op.inputs[2],
                                output_shape = op.inputs[0].shape[1:4].as_list(),
                                det_sz = op.get_attr('det_sz'), 
                                det_off = op.get_attr('det_off'),
                                dsd = op.get_attr('dsd'),
                                dso = op.get_attr('dso'),
                                type_projector = op.get_attr('type_projector'))
    
    return g, None, None



# In[6]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'ReconNet'])

