#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ctypes import *
import os
import numpy as np
import scipy.ndimage
import time
import spams


# In[2]:


if __name__ == '__main__':
    cmodule = cdll.LoadLibrary('./libPriorFunctionSolver.so')
else:
    cmodule = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libPriorFunctionSolver.so'))


# In[3]:


def mySVD(x):
    m = x.shape[0]
    n = x.shape[1]
    
    u = np.copy(x, 'C')
    u = u.astype(np.float32)
    s = np.zeros([n], np.float32)
    v = np.zeros([n,n], np.float32)
    
    cmodule.svdcmp_host(u.ctypes.data_as(POINTER(c_float)), 
                        s.ctypes.data_as(POINTER(c_float)), 
                        v.ctypes.data_as(POINTER(c_float)), 
                        c_int(m), c_int(n))
    
    return u, s, v

def mySVDRecon(u, s, v):
    m = u.shape[0]
    n = v.shape[0]
    
    u = u.astype(np.float32)
    s = s.astype(np.float32)
    v = v.astype(np.float32)
    
    cmodule.svdrecon_host(u.ctypes.data_as(POINTER(c_float)), 
                          s.ctypes.data_as(POINTER(c_float)), 
                          v.ctypes.data_as(POINTER(c_float)), 
                          c_int(m), c_int(n))
    
    return u

def myNuclearNormSoftThresh(x, mu, p):
    m = x.shape[0]
    n = x.shape[1]
    
    x = x.astype(np.float32)
    
    cmodule.NuclearNormSoftThresh_host(x.ctypes.data_as(POINTER(c_float)), 
                                       c_int(m), c_int(n), c_float(mu), c_float(p))
    
    return x


# In[4]:


def cSetDevice(device):
    return cmodule.cSetDevice(c_int(device))


# In[5]:


def cNuclearNormSoftThresh(x, mu, p):
    batch = x.shape[0]
    m = x.shape[1]
    n = x.shape[2]
    
    x = x.astype(np.float32)
    s = np.zeros([batch, n], np.float32)
    
    cmodule.cNuclearNormSoftThresh(x.ctypes.data_as(POINTER(c_float)), 
                                   s.ctypes.data_as(POINTER(c_float)), 
                                   c_int(m), c_int(n), c_int(batch), c_float(mu), c_float(p))
    
    return x, s


# In[6]:


def cOMP(x, D, L, thresh=1e-16):
    a = np.zeros(x.shape, np.float32)
    r = np.zeros(x.shape, np.float32)
    
    D = np.copy(np.reshape(D, [D.shape[0], -1]), 'C')
    DtD = np.copy(D @ D.T, 'C')
    
    cnt = x.shape[0]
    n = int(x.size / cnt)
    d = D.shape[0]
    
#     print (x.shape, D.shape, DtD.shape, L, n, d, cnt)
    
    cmodule.cOrthogonalMatchingPursuit(a.ctypes.data_as(POINTER(c_float)), 
                                       r.ctypes.data_as(POINTER(c_float)), 
                                       x.ctypes.data_as(POINTER(c_float)), 
                                       D.ctypes.data_as(POINTER(c_float)), 
                                       DtD.ctypes.data_as(POINTER(c_float)), 
                                       c_int(L), 
                                       c_int(n), 
                                       c_int(d), 
                                       c_int(cnt), 
                                       c_float(thresh))
    
    return a, r


# In[7]:


def myOMP(v, D, L):
    approx = np.zeros_like(v)
    residue = np.zeros_like(v)
    
    for n in range(v.shape[1]):
        inds = []
        r = np.copy(v[:, [n]])

        for i in range(L):
            # find max ind
            inds.append(np.argmax(np.abs(r.T @ D)))

            # construct current dictionary
            Phi = np.concatenate([D[:, [i]] for i in inds], -1)

            # new signal estimate
            x = np.linalg.lstsq(Phi, v[:, [n]], None)[0]

    #         print (x.T)

            # new residue and estimate
            a = Phi @ x
            r = v[:, [n]] - a
        
        approx[:, [n]] = a
        residue[:, [n]] = r
    
    return approx, residue


# In[8]:


def cTVSQS2D(x, eps = 1e-8):
    x = x.astype(np.float32)
    
    s1 = np.zeros(x.shape, np.float32)
    s2 = np.zeros(x.shape, np.float32)
    var = np.zeros(x.shape, np.float32)
    
    cmodule.cTVSQS2D(s1.ctypes.data_as(POINTER(c_float)), 
                     s2.ctypes.data_as(POINTER(c_float)), 
                     var.ctypes.data_as(POINTER(c_float)), 
                     x.ctypes.data_as(POINTER(c_float)),
                     c_int(x.shape[0]),
                     c_int(x.shape[1]),
                     c_int(x.shape[2]),
                     c_int(x.shape[3]),
                     c_float(eps))
    
    return s1, s2, var


# In[9]:


def cTVSQS3D(x, eps = 1e-8):
    x = x.astype(np.float32)
    
    s1 = np.zeros(x.shape, np.float32)
    s2 = np.zeros(x.shape, np.float32)
    var = np.zeros(x.shape, np.float32)
    
    cmodule.cTVSQS3D(s1.ctypes.data_as(POINTER(c_float)), 
                     s2.ctypes.data_as(POINTER(c_float)), 
                     var.ctypes.data_as(POINTER(c_float)), 
                     x.ctypes.data_as(POINTER(c_float)),
                     c_int(x.shape[0]),
                     c_int(x.shape[1]),
                     c_int(x.shape[2]),
                     c_int(x.shape[3]),
                     c_int(x.shape[4]),
                     c_float(eps))
    
    return s1, s2, var


# In[10]:


def MakePatch2D(x, patchsize, step, coords = None, randomStep=False):
    if coords is None:
        xs = np.arange(0, x.shape[1] - patchsize[0], step[0])
        if xs[-1] + patchsize[0] < x.shape[1]:
            xs = np.concatenate((xs, [x.shape[1] - patchsize[0]]))
        ys = np.arange(0, x.shape[2] - patchsize[1], step[1])
        if ys[-1] + patchsize[1] < x.shape[2]:
            ys = np.concatenate((ys, [x.shape[2] - patchsize[1]]))
        
        coords = np.meshgrid(xs, ys)
        
        if randomStep:
            coords[0][:, 1:-1] += np.random.randint(-int(step[0]/2), int(step[0]/2)+1, coords[0][:, 1:-1].shape)
            coords[1][1:-1, :] += np.random.randint(-int(step[0]/2), int(step[0]/2)+1, coords[1][1:-1, :].shape)
            coords[0][coords[0] > x.shape[1] - patchsize[0]] = x.shape[1] - patchsize[0]
            coords[1][coords[1] > x.shape[2] - patchsize[1]] = x.shape[2] - patchsize[1]
    
    patches = np.zeros([coords[0].size, 
                        x.shape[0], patchsize[0], patchsize[1], x.shape[3]], np.float32)
    
    i = 0
    for ix in range(coords[0].shape[0]):
        for iy in range(coords[0].shape[1]):
            patches[i, ...] = x[:, coords[0][ix,iy]:coords[0][ix,iy]+patchsize[0], 
                                coords[1][ix,iy]:coords[1][ix,iy]+patchsize[1], :]
            i += 1
    
    return patches, coords

def MakePatch2DTranspose(patches, coords, imgshape):
    x = np.zeros(imgshape, np.float32)
    
    i = 0
    for ix in range(coords[0].shape[0]):
        for iy in range(coords[0].shape[1]):
            x[:, coords[0][ix,iy]:coords[0][ix,iy]+patches.shape[2], 
              coords[1][ix,iy]:coords[1][ix,iy]+patches.shape[3], :] += patches[i,...]
            i += 1
    
    return x


# In[11]:


def NuclearNormShrinkAndSQS(x, patchsize, step, mu, p, randomStep = False):
    # the shrink step
    patches, coords = MakePatch2D(x, patchsize, step, None, randomStep)
    y = np.reshape(patches, [-1, patchsize[0] * patchsize[1], patches.shape[-1]])
    y, s = cNuclearNormSoftThresh(y, mu, p)
    y = np.reshape(y, patches.shape)
    
    # the sqs step
    s1 = MakePatch2DTranspose(patches - y, coords, x.shape)
    s2 = MakePatch2DTranspose(np.ones_like(patches), coords, x.shape)
    
    return s1, s2, s

def CalcNulcearNorm(s, mu, p):
    th = mu**(1 / (2-p))
    s = np.abs(s)
    
    s1 = s**2 / (2 * mu)
    s2 = s**p / p - (1 / p - 0.5) * th
    
    loss = np.zeros_like(s)
    loss[s < th] = s1[s < th]
    loss[s >= th] = s2[s >= th]
    
    return np.sum(loss)


# In[12]:


def softThresh(s, mu, p):
    abss = np.abs(s)
    y = abss - mu * abss ** (p-1)
    y[y < 0] = 0
    
    return y * np.sign(s)


# In[8]:


def ompSQS(img, D, L, patchsize, step, randomStep = False, coords = None, gpu=True):
    patches, coords = MakePatch2D(img, patchsize, step, coords, randomStep)
    
    x = np.reshape(patches, [patches.shape[0] * patches.shape[1], -1])
#     mx = np.tile(np.mean(x, -1)[..., np.newaxis], (1, x.shape[-1]))
#     x -= mx
    
    if gpu:
        # gpu routine
        x = np.copy(x, 'C')
        D = np.copy(D, 'C')
        x, _ = cOMP(x, D, L)
    else:
        # spams routine
        x = np.asfortranarray(x.T, np.float32)
        D = np.asfortranarray(D.T, np.float32)
        coefs = spams.omp(x, D, L)
        x = (D @ coefs).T
    
#     x = np.reshape(x + mx, patches.shape)
    x = np.reshape(x, patches.shape)
    
    d1 = MakePatch2DTranspose(patches - x, coords, img.shape)
    d2 = MakePatch2DTranspose(np.ones_like(patches), coords, img.shape)
    
    return d1, d2, np.sqrt(np.sum((patches - x)**2))

def imgOMP(img, D, L, patchsize, step, randomStep = False, coords = None, gpu=True):
    patches, coords = MakePatch2D(img, patchsize, step, coords, randomStep)
    
    x = np.reshape(patches, [patches.shape[0] * patches.shape[1], -1])
    mx = np.tile(np.mean(x, -1)[..., np.newaxis], (1, x.shape[-1]))
    x -= mx
    
    if gpu:
        # gpu routine
        x = np.copy(x, 'C')
        D = np.copy(D, 'C')
        x, _ = cOMP(x, D, L)
    else:
        # spams routine
        x = np.asfortranarray(x.T, np.float32)
        D = np.asfortranarray(D.T, np.float32)
        coefs = spams.omp(x, D, L)
        x = (D @ coefs).T
    
    x = np.reshape(x + mx, patches.shape)
    
    d1 = MakePatch2DTranspose(x, coords, img.shape)
    d2 = MakePatch2DTranspose(np.ones_like(x), coords, img.shape)
    
    return d1 / d2


# In[9]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'PriorFunctionSolver'])


# In[ ]:




