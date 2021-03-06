#############################
##
## LightFields.py
## Define LightField class
##
## Athanasios Athanassiadis
## Sept. 2013
##
## MIT License
##
#############################
from __future__ import division
import sys
import numpy as np
import scipy.ndimage as ndi
# import pylab as pl

from MatchFeats import get_dxdy


class LFSizeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class LFShapeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class LFClassError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class LFGenError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LightField(object):
    """
    Skeleton class for Light Field objects

    outlines basic variables and functions that subclasses should implement

    """

    def __init__(self):
        self.data = None
        self.depth_field = None
        self.nu = 0
        self.nv = 0
        self.ncolor = 0
        self.fpix = 0
        self.aperture = 0
        self.acx = 0
        self.acy = 0
        self.changed = True

    def set_data(self, d):
        self.changed=True
        np.copyto(self.data, d)

    def set_focus(self, f):
        self.changed=True
        self.fpix = f

    def set_aperture(self, ap):
        self.changed=True
        self.aperture=ap

    def get_data(self):
        return self.data.copy()

    def get_depth(self):
        return self.depth_field

    def get_params(self):
        p = {"focus" : self.fpix,
             "aperture" : self.aperture,
             "xcenter" : self.acx,
             "ycenter" : self.acy,
             "nv": self.nv,
             "nu": self.nu,
             "ncolor": self.ncolor}

        return p

    def dump(self, mode):
        """ this should dump data based on mode : either spatially organized or angularly organized """
        pass

    def calc_depth(self):
        """ calculate the depth field for a given image """

    def render(self):
        """ render an image based on current parameters """
        pass

    def copy(self, newLF):
        """
        set up the properties of the new lightfield
        then copy in the data (use np.copyto)
        """
        pass


class RLightField(LightField):
    """
    RectLightField - Light Field from samples on a rectangular grid

    parameters:
        data
        nu,nv
        nx,ny
        ncolor
        mask
        aperture
        acx,acy
        fpix

    """

    def __init__(self, data=None, nx=1, ny=1, nu=1, nv=1, colors=1,
                 mask=None, aperture=1, fpix=0, acx=None, acy=None):

        super(RLightField,self).__init__()

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise LFClassError("Please supply data in numpy ndarray format. Light field not initialized.")

            ny,nx,nv,nu,colors = data.shape

        ## image resolution
        self.nv = nv
        self.nu = nu

        ## array size
        self.ny = ny
        self.nx = nx

        ## number of color channels (grayscale vs rgb)
        self.ncolor = colors

        ## set lightfield data
        ## 5D array with each dimension indexed by:
        ## y_cam, x_cam, i, j, rgb/gray
        self.data = np.empty([ny,nx,nv,nu,colors],
                             order="C",
                             dtype=np.uint8)

        if data is not None:
            self.set_data(data)

        ## 2D mask for which camera views to use
        ## default to all on
        if mask is None:
            self.mask = np.ones([ny, nx], dtype=np.float32)

        if type(mask) == np.ndarray and mask.shape != (ny,nx):
            sys.stdout.write('(W) SGLightfield: Could not apply given mask to this lightfield. Reverting to default mask.\n')
            sys.stdout.flush()
            self.mask = np.ones([ny,nx], dtype=np.float32)

        ## set aperture center
        if acx is None:
            self.acx = nx//2
        if acy is None:
            self.acy = ny//2

        self.aperture = aperture
        self.set_aperture(aperture) ## set the depth of field by modifying aperture; default to max
        self.fpix = fpix
        self.set_focus(fpix) 	  ## set the focus; default to 0

        ## changed tracks whether or not we need to
        ## re-render before outputting the reconstructed image
        self.changed = True

        ## data structure to hold output lightfield and image
        self.LFout = np.empty([ny,nx,nv,nu,colors], order="C", dtype=np.float32)
        self.output = np.empty([nv,nu,colors], order="C", dtype=np.float32)

    def set_data(self, data):
        """ set the lightfield data """

        if data.ndim != 5:
            raise LFSizeError("Expected data to have 5 dimensions, instead found %d. set_data() aborted." % data.ndim)
        elif data.shape != self.data.shape:
            raise LFShapeError("Expected data to have shape %s and instead found %s. set_data() aborted." % (str(self.data.shape), str(data.shape)))
        else:
            self.changed = True

            if data.max() <= 1:
                data *= 255
            if data.dtype is not np.dtype('uint8'):
                data = data.astype(np.uint8)

            np.copyto(self.data, data)

    def set_focus(self, foc):
        self.changed = True
        self.fpix = foc

    def set_center(self, acx,acy):
        self.changed = True
        self.acx = acx
        self.acy = acy
        self.set_aperture(self.aperture)

    def set_aperture(self, ap, constricted=True):
        self.changed = True
        self.aperture = max(ap, 0.5)				## lower bound at .5

        if constricted:
            self.aperture = min(ap, self.nx, self.ny)	## upper bound at shortest edge length

        ## set a circular aperture
        YY,XX = np.mgrid[:self.ny, :self.nx]
        x = XX - self.acx
        y = YY - self.acy

        mask = (np.sqrt(x**2 + y**2) < self.aperture/2).astype(np.float32)
        self.set_mask(mask)

    def set_mask(self, mask):
        self.changed = True
        np.copyto(self.mask, mask)

    def multiply_mask(self, mult):
        self.changed = True
        self.mask *= mult

    def get_data(self):
        """ get the raw data """
        return self.data.copy()

    def get_mask(self):
        return self.mask

    def get_params(self):
        p = {"focus" : self.fpix,
             "aperture" : self.aperture,
             "nv": self.nv,
             "xcenter" : self.acx,
             "ycenter" : self.acy,
             "nu": self.nu,
             "ny": self.ny,
             "nx": self.nx,
             "ncolor": self.ncolor}

        return p

    def dump(self, mode=1, spacing=0):
        """
        dump the light field data as one giant image in one of two formats:
        1 - spatial (each subframe shows an individual camera capture)
        2 - angular (each subframe shows all of the light entering from a given direction)

        """

        ## spatial dump
        if mode is 1:
            outimage = np.zeros((self.ny*self.nv + spacing*(self.ny-1),
                                 self.nx*self.nu + spacing*(self.nx-1),
                                 self.ncolor), order="C", dtype=np.float32)
            for y in xrange(self.ny):
                for x in xrange(self.nx):
                    outimage[y*(self.nv+spacing):(y+1)*self.nv + spacing*y,
                             x*(self.nu+spacing):(x+1)*self.nu + spacing*x, :] = self.data[y,x]

        ## angular dump

        ## this is the slow way - cycle through nu/nv
        #elif mode is 2:
        #    outimage = np.zeros((self.ny*self.nv + spacing*(self.nv-1),
        #                         self.nx*self.nu + spacing*(self.nu-1),
        #                         self.ncolor), order="C", dtype=np.float32)
        #    for v in xrange(self.nv):
        #        for u in xrange(self.nu):
        #            outimage[v*(self.ny+spacing):(v+1)*self.ny + spacing*v,
        #                     u*(self.nx+spacing):(u+1)*self.nx + spacing*u, :] = self.data[:,:,v,u,:]

        ## this is the faster way - assuming that there are more pixels than cameras
        elif mode is 2:
            outimage = np.zeros((self.ny*self.nv + spacing*(self.nv-1),
                                 self.nx*self.nu + spacing*(self.nu-1),
                                 self.ncolor), order="C", dtype=np.float32)

            for y in xrange(self.ny):
                for x in xrange(self.nx):
                    outimage[y:outimage.shape[0]:self.ny + spacing,
                             x:outimage.shape[1]:self.nx + spacing, :] = self.data[y,x]

        else:
            raise LFGenError("Only two dump modes supported. Dump aborted.")

        return outimage.copy()

    def calc_depth(self):
        """ calculate the depth field """

        pass

    def render(self, outType='mean', outputLF=False):

        ## if we need to update the rendered image, then do so
        if self.changed:

            ## get grid sampling points for the images
            VV,UU = np.mgrid[:self.nv, :self.nu]

            ## set normalized weights
            w = self.mask

            if not(outType in ['median','majority']):
                w /= self.mask.sum()

            ## clear any old renderings
            self.LFout = np.empty([self.ny,self.nx, self.nv, self.nu, self.ncolor], order="C", dtype=np.float32)

            I = np.empty([self.nv,self.nu, self.ncolor], order="C", dtype=np.float32)

            ## cycle through the different camera positions and colors
            for y in xrange(self.ny):
                for x in xrange(self.nx):
                    for c in xrange(self.ncolor):
                        ## map coordinates with linear spline interpolation
                        if w[y,x] != 0:
                            U1 = UU + self.fpix * (x - self.acx)
                            V1 = VV - self.fpix * (y - self.acy)
                            I[:,:,c] = ndi.interpolation.map_coordinates(self.data[y,x,:,:,c], [V1,U1],
                                                                         order=1, mode='constant')
                        else:
                            I[:,:,c] = 0


                    ## copy resampled image to output LightField
                    self.LFout[y,x] =  (I * w[y,x]).copy()
        ## generate output image based on desired reconstruction method
        if np.char.lower(outType) == 'median':
            ## get grid sampling points for the images
            VV,UU = np.mgrid[:self.nv, :self.nu]

            medianVote = lambda V,U,C: np.median(self.LFout[:,:,V,U,c])
            vf = np.vectorize(medianVote, excluded={'c'})
            for c in range(3):
                self.output[:,:,c] = vf(VV.flat, UU.flat, c).reshape(self.nv,self.nu)

        if np.char.lower(outType) == 'majority':
            ## get grid sampling points for the images
            VV,UU = np.mgrid[:self.nv, :self.nu]
            def popVote(V,U,c):
                h,b = np.histogram(self.LFout[:,:,V,U,c].flat, bins=np.arange(-.5,256.5,1))
                b = np.arange(256)
                if h.max() <= self.mask.sum()//2:
                    out = 0
                else:
                    out = b[h==h.max()][0]

                return out


            vfunc = np.vectorize(popVote, excluded={'c'})
            for c in range(3):
                self.output[:,:,c] = vfunc(VV.flat, UU.flat, c).reshape(self.nv,self.nu)

        else:
            self.output = self.LFout.sum((0,1))


            self.changed = False

        if outputLF:
            return self.output.copy(), self.LFout.copy()
        else:
            return self.output.copy()

    def copy(self, newLF):
        """ copy all parameters and data to a new RLightField object """

        if not isinstance(newLF, self.__class__):
            raise LFClassError("Cannot copy data from type %s to type %s. copy() aborted." % (self.__class__,type(newLF)))

        else:
            newLF.changed = True
            newLF.nv = self.nv
            newLF.nu = self.nu
            newLF.ny = self.ny
            newLF.nx = self.nx
            newLF.ncolor = self.ncolor
            newLF.mask = np.empty_like(self.mask)
            np.copyto(newLF.mask, self.mask)
            newLF.data = np.empty_like(self.data)
            np.copyto(newLF.data, self.data)
            newLF.aperture = self.aperture
            newLF.fpix= self.fpix
            #newLF.spacing = self.spacing
            newLF.output = np.empty_like(self.output)
            np.copyto(newLF.output, self.output)

