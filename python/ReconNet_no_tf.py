#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ctypes import *
import os
import numpy as np
import re
import scipy.ndimage


# In[2]:


if __name__ == '__main__':
    cmodule = cdll.LoadLibrary('./libReconNet4D_no_tf.so')
else:
    cmodule = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libReconNet4D_no_tf.so'))


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
        for k in params:
            setattr(self, k, params[k])
#         for varname in dir(self):
#             if varname in params:
#                 setattr(self, varname, params[varname])
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
    
    def cPixelBackprojection4d(self, sino, angles = None, img_shape = None, type_projector = 0):
        sino = sino.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, self.nz]
        
        img = np.zeros([sino.shape[0], img_shape[0], img_shape[1], img_shape[2], sino.shape[4]], np.float32)
        
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
    
    def cPixelBackprojection3d(self, sino, angles = None, img_shape = None, type_projector = 0):
        if img_shape is None:
            img_shape = [self.nx, self.ny, 1]
        
        img = self.cPixelBackprojection4d(sino[..., np.newaxis, :], angles, img_shape, type_projector)        
        return np.squeeze(img, -2)
    
    def cDDFanProjection4d(self, img, angles = None, sino_shape = None, type_projector = 0):
        img = img.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if sino_shape is None:
            sino_shape = [self.nu, len(angles), self.nv]
        
        sino = np.zeros([img.shape[0], sino_shape[0], sino_shape[1], sino_shape[2], img.shape[4]], np.float32)
        
        cmodule.cDistanceDrivenFanProjection(
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
    
    def cDDFanProjection3d(self, img, angles = None, sino_shape = None, type_projector = 0):
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if sino_shape is None:
            sino_shape = [self.nu, len(angles), 1]
        
        sino = self.cDDFanProjection4d(img[..., np.newaxis, :], angles, sino_shape, type_projector)
        return np.squeeze(sino, -2)
    
    def cDDFanBackprojection4d(self, sino, angles = None, img_shape = None, type_projector = 0):
        sino = sino.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, self.nz]
        
        img = np.zeros([sino.shape[0], img_shape[0], img_shape[1], img_shape[2], sino.shape[4]], np.float32)
        
        cmodule.cDistanceDrivenFanBackprojection(
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
    
    def cDDFanBackprojection3d(self, sino, angles = None, img_shape = None, type_projector = 0):
        if img_shape is None:
            img_shape = [self.nx, self.ny, 1]
        
        img = self.cDDFanBackprojection4d(sino[..., np.newaxis, :], angles, img_shape, type_projector)        
        return img[..., 0, :]
    
    def cDDFanNormImg4d(self, sino, wsino = None, angles = None, img_shape = None, type_projector = 0):
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
            
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, self.nz]

        fp = self.cDDFanProjection4d(np.ones([sino.shape[0], 
                                              img_shape[0], 
                                              img_shape[1], 
                                              img_shape[2], 
                                              sino.shape[4]], np.float32), 
                                     angles, sino.shape[1:4], type_projector)
        bp = self.cDDFanBackprojection4d(fp * wsino, angles, img_shape, type_projector)
        return bp
    
    def cDDFanNormImg3d(self, sino, wsino = None, angles = None, img_shape = None, type_projector = 0):
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, 1]
            
        normImg = self.cDDFanNormImg4d(sino[..., np.newaxis, :], 
                                       wsino[..., np.newaxis, :], 
                                       angles, img_shape, type_projector)
        return normImg[..., 0, :]
    
    def cDDTomoProjection4d(self, img, detCenter, src):
        img = img.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = np.array([[1,0,0]]*detCenter.shape[0], np.float32)
        detV = np.array([[0,1,0]]*detCenter.shape[0], np.float32)
        src = src.astype(np.float32)
        
        prj = np.zeros([img.shape[0], self.nu, detCenter.shape[0], self.nv, img.shape[-1]], np.float32)
        cmodule.cDistanceDrivenTomoProjection(
            prj.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            detCenter.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), c_int(img.shape[-1]), 
            c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return prj
    
    def cDDTomoBackprojection4d(self, prj, detCenter, src):
        prj = prj.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        src = src.astype(np.float32)
        
        img = np.zeros([prj.shape[0], self.nx, self.ny, self.nz, prj.shape[-1]], np.float32)
        cmodule.cDistanceDrivenTomoBackprojection(
            img.ctypes.data_as(POINTER(c_float)), 
            prj.ctypes.data_as(POINTER(c_float)), 
            detCenter.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), c_int(img.shape[-1]), 
            c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return img
    
    def cDDTomoNormImg4d(self, prj, detCenter, src, mask = None):        
        img = np.ones([prj.shape[0], self.nx, self.ny, self.nz, prj.shape[-1]], np.float32)
        if mask is None:
            mask = np.ones_like(img)
        fp = self.cDDTomoProjection4d(img * mask, detCenter, src)
        bp = self.cDDTomoBackprojection4d(fp, detCenter, src) * mask
        
        return bp
    
    def cSiddonFanProjection4d(self, img, angles = None, sino_shape = None, type_projector = 0):
        img = img.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if sino_shape is None:
            sino_shape = [self.nu, len(angles), self.nv]
        
        sino = np.zeros([img.shape[0], sino_shape[0], sino_shape[1], sino_shape[2], img.shape[4]], np.float32)
        
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
    
    def cSiddonFanProjection3d(self, img, angles = None, sino_shape = None, type_projector = 0):
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if sino_shape is None:
            sino_shape = [self.nu, len(angles), 1]
        
        sino = self.cSiddonFanProjection4d(img[..., np.newaxis, :], angles, sino_shape, type_projector)
        return np.squeeze(sino, -2)
    
    def cSiddonFanBackprojection4d(self, sino, angles = None, img_shape = None, type_projector = 0):
        sino = sino.astype(np.float32)
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, self.nz]
        
        img = np.zeros([sino.shape[0], img_shape[0], img_shape[1], img_shape[2], sino.shape[4]], np.float32)
        
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
    
    def cSiddonFanBackprojection3d(self, sino, angles = None, img_shape = None, type_projector = 0):
        if img_shape is None:
            img_shape = [self.nx, self.ny, 1]
        
        img = self.cSiddonFanBackprojection4d(sino[..., np.newaxis, :], angles, img_shape, type_projector)        
        return img[..., 0, :]
    
    def cSiddonFanNormImg4d(self, sino, wsino = None, angles = None, img_shape = None, type_projector = 0):
        if angles is None:
            angles = np.array(self.angles, np.float32)
        else:
            angles = angles.astype(np.float32)
            
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, self.nz]

        fp = self.cSiddonFanProjection4d(np.ones([sino.shape[0], 
                                                  img_shape[0], 
                                                  img_shape[1], 
                                                  img_shape[2], 
                                                  sino.shape[4]], np.float32), 
                                         angles, sino.shape[1:4], type_projector)
        bp = self.cSiddonFanBackprojection4d(fp * wsino, angles, img_shape, type_projector)
        return bp
    
    def cSiddonFanNormImg3d(self, sino, wsino = None, angles = None, img_shape = None, type_projector = 0):
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
        
        if img_shape is None:
            img_shape = [self.nx, self.ny, 1]
            
        normImg = self.cSiddonFanNormImg4d(sino[..., np.newaxis, :], 
                                           wsino[..., np.newaxis, :], 
                                           angles, img_shape, type_projector)
        return normImg[..., 0, :]
    
    def cSiddonFanOSSQS4d(self, nSubsets, img, sino, normImg, wsino = None, angles = None, order = None, 
                          lb = None, ub = None, type_projector = 0, lam = 1):
        img = img.astype(np.float32)
        sino = sino.astype(np.float32)
        normImg = normImg.astype(np.float32)
        
        if lb is None:
            lb = np.finfo(np.float32).min
        if ub is None:
            ub = np.finfo(np.float32).max
        
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
            c_float(lb), c_float(ub),
            c_int(nviewPerSubset), c_int(nSubsets),
            c_int(img.shape[0]), c_int(img.shape[4]), c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]),
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_int(sino.shape[1]), c_int(sino.shape[2]), c_int(sino.shape[3]),
            c_float(self.da), c_float(self.dv), c_float(self.off_a), c_float(self.off_v),
            c_float(self.dsd), c_float(self.dso),
            c_int(type_projector), c_float(lam))
        
        return res
    
    def cSiddonFanOSSQS3d(self, nSubsets, img, sino, normImg, wsino = None, angles = None, order = None, 
                          lb = None, ub = None, type_projector = 0, lam = 1):
        if wsino is None:
            wsino = np.ones(sino.shape, np.float32)
        else:
            wsino = wsino.astype(np.float32)
        
        res = self.cSiddonFanOSSQS4d(nSubsets, img[..., np.newaxis, :], sino[..., np.newaxis, :], 
                                     normImg[..., np.newaxis, :], wsino[..., np.newaxis, :], 
                                     angles, order, lb, ub, type_projector, lam)
        
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
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
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
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return img
    
    def cSiddonConeProjectionAbitrary4d(self, img, detCenter, detU, detV, src):
        img = img.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = detU.astype(np.float32)
        detV = detV.astype(np.float32)
        src = src.astype(np.float32)
        
        prj = np.zeros([img.shape[0], self.nu, detCenter.shape[0], self.nv, img.shape[-1]], np.float32)
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
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
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
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        return img
    
    def cSiddonConeNormImgAbitrary4d(self, prj, detCenter, detU, detV, src, mask = None):        
        img = np.ones([prj.shape[0], self.nx, self.ny, self.nz, prj.shape[-1]], np.float32)
        if mask is None:
            mask = np.ones_like(img)
        fp = self.cSiddonConeProjectionAbitrary4d(img * mask, detCenter, detU, detV, src)
        bp = self.cSiddonConeBackprojectionAbitrary4d(fp, detCenter, detU, detV, src) * mask
        
        return bp
    
    def cSiddonConeAbitraryOSSQS4d(self, nSubsets, img, prj, normImg, 
                                   detCenter, detU, detV, src, mask = None, order = None, lb = None, ub = None, lam = 1):
        img = img.astype(np.float32)
        prj = prj.astype(np.float32)
        normImg = normImg.astype(np.float32)
        detCenter = detCenter.astype(np.float32)
        detU = detU.astype(np.float32)
        detV = detV.astype(np.float32)
        src = src.astype(np.float32)
        
        if mask is None:
            mask = np.ones(img.shape, np.float32)
        else:
            mask = mask.astype(np.float32)
        
        nviewPerSubset = int(np.ceil(prj.shape[2] / float(nSubsets)))
        
        if lb is None:
            lb = np.finfo(np.float32).min
        if ub is None:
            ub = np.finfo(np.float32).max
        
        if order is not None:
            prj = np.copy(prj[:,:,order,:,:], 'C')
            detCenter = np.copy(detCenter[order,...], 'C')
            detU = np.copy(detU[order,...], 'C')
            detV = np.copy(detV[order,...], 'C')
            src = np.copy(src[order,...],'C')
        
        res = np.zeros(img.shape, np.float32)
        
        cmodule.cOSSQSConeAbitrary(
            res.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            prj.ctypes.data_as(POINTER(c_float)), 
            normImg.ctypes.data_as(POINTER(c_float)), 
            mask.ctypes.data_as(POINTER(c_float)),
            detCenter.ctypes.data_as(POINTER(c_float)), 
            detU.ctypes.data_as(POINTER(c_float)), 
            detV.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)), 
            c_float(lb), c_float(ub),
            c_int(nviewPerSubset), c_int(nSubsets), 
            c_int(img.shape[0]), c_int(img.shape[4]), c_int(img.shape[1]), c_int(img.shape[2]), c_int(img.shape[3]),
            c_float(self.dx), c_float(self.dy), c_float(self.dz), 
            c_float(self.cx), c_float(self.cy), c_float(self.cz), 
            c_int(prj.shape[1]), c_int(prj.shape[2]), c_int(prj.shape[3]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v), 
            c_int(2), c_float(lam))
        
        return res
    
    def cTVSQS2d(self, img):
        img = img.astype(np.float32)
        
        if img.ndim != 4 or img.shape[0] != 1 or img.shape[-1] != 1:
            raise ValueError('img.shape must be [1, _, _, 1]');
        
        grad = np.zeros(img.shape, np.float32)
        norm = np.zeros(img.shape, np.float32)
        
        cmodule.cTVSQS2D.restype = c_float
        tv = cmodule.cTVSQS2D(
            grad.ctypes.data_as(POINTER(c_float)), 
            norm.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            c_int(img.shape[1]), c_int(img.shape[2]))
        
        return grad, norm, tv
    
    def get_img_shape(self):
        return [self.nx, self.ny, self.nz]
    
    def get_sino_shape(self):
        return [self.nu, self.rotview, self.nv]


# In[4]:


def ParkerWeighting(reconNet):
    gamma = (reconNet.angles[-1] - np.pi) / 2
    delta = reconNet.da * reconNet.nu / 2
    # if gamma < delta:
    #     gamma = delta
    alphas = ((np.arange(reconNet.nu) - (reconNet.nu - 1) / 2) - reconNet.off_a) * reconNet.da
    weights = []
    for beta in reconNet.angles:
        w = np.zeros([reconNet.nu], np.float32)
        res1 = (np.sin(np.pi / 4 * beta / (gamma - alphas)))**2
        res2 = (np.sin(np.pi / 4 * (np.pi + 2 * gamma - beta) / (gamma + alphas)))**2
        i1 = np.where(beta < 2 * gamma - 2 * alphas)[0]
        i2 = np.where(beta <= np.pi - 2 * alphas)[0]
        w = res2
        w[i2] = 1
        w[i1] = res1[i1]
        weights.append(w)
    weights = np.transpose(np.array(weights))[np.newaxis, ..., np.newaxis]
    
    return weights


# In[5]:


def RiessWeighting(reconNet, sigma = 30, theta = 10 * np.pi / 180):
    gamma = (reconNet.angles[-1] - np.pi) / 2
    delta = reconNet.da * reconNet.nu / 2
    # if gamma < delta:
    #     gamma = delta
    alphas = ((np.arange(reconNet.nu) - (reconNet.nu - 1) / 2) - reconNet.off_a) * reconNet.da
    weights = []
    for beta in reconNet.angles:
        w = np.ones([reconNet.nu], np.float32)
        res1 = (np.sin(np.pi / 4 * beta / (gamma - alphas)))**2
        res2 = (np.sin(np.pi / 4 * (np.pi + 2 * gamma - beta) / (gamma + alphas)))**2
        i1 = np.where(beta < 2 * gamma - 2 * alphas)[0]
        i2 = np.where(beta < -2 * gamma + 2 * alphas)[0]
        i3 = np.where(beta > np.pi - 2 * alphas)[0]
        i4 = np.where(beta > np.pi + 2 * gamma + 2 * alphas)[0]
        w[i4] = 2 - res2[i4]
        w[i3] = res2[i3]
        w[i2] = 2 - res1[i2]
        w[i1] = res1[i1]

        # gaussian smoothing
        wSmooth = scipy.ndimage.gaussian_filter1d(w, sigma, mode='nearest')
        ind = np.where(np.all((beta > theta, beta < np.pi + 2 * gamma - theta)))
        wSmooth[ind] = w[ind]

        weights.append(wSmooth)
    weights = np.transpose(np.array(weights))[np.newaxis, ..., np.newaxis]
    
    return weights


# In[1]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'ReconNet_no_tf'])


# In[ ]:




