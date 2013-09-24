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


class RLightField(object):
	"""
	RectLightField - Light Field from samples on a rectangular grid

	parameters:
		nu,nv
		nx,ny
		ncolor
		mask
		aperture
		acx,acy
		fpix
		spacing


	"""

	def __init__(self, nx=1, ny=1, nu=1, nv=1, colors=1, mask=None, aperture=-1, fpix=0, spac=None, acx=None, acy=None):

		## image resolution
		self.nv = nv
		self.nu = nu

		## array size
		self.ny = ny
		self.nx = nx
		
		## number of color channels (grayscale vs rgb)
		self.ncolor = colors

		## 2D mask for which camera views to use
		## default to all on
		if mask is None:
			self.mask = np.ones([ny,nx], dtype=np.float32)

		if type(mask)==np.ndarray and mask.shape != (ny,nx):
			sys.stdout.write('(W) SGLightfield: Could not apply given mask to this lightfield. Reverting to default mask.\n')
			sys.stdout.flush()
			self.mask = np.ones([ny,nx], dtype=np.float32)
		
		## set lightfield data
		## 5D array with each dimension indexed by:
		## y_cam, x_cam, i, j, rgb/gray
		self.data = np.zeros([ny,nx,nv,nu,colors],
							 order="C",
							 dtype=np.float32)

		## set aperture center
		if acx is None:
			self.acx = nx//2
		if acy is None:
			self.acy = ny//2


		self.aperture = aperture
		self.set_aperture(aperture) ## set the depth of field by modifying aperture; default to max
		self.fpix = fpix
		self.refocus(fpix) 	  ## set the focus; default to 

		## [optional] real-world grid spacing in (dx,dy) format
		self.spacing = spac

		## changed tracks whether or not we need to 
		## re-render before outputting the reconstructed image
		self.changed = True

		## data structure to hold output lightfield and image
		self.LFOut = np.empty([ny,nx,nv,nu,colors], order="C", dtype=np.float32)
		self.output = np.empty([nv,nu,colors], order="C", dtype=np.float32)


	def get_data(self):
		''' get the raw data '''
		return self.data.copy()


	def get_mask(self):
		return self.mask


	def get_params(self):
		p = {"focus" : self.focus,
			 "aperture" : self.aperture,
			 "nv" : self.nv,
			 "nu" : self.nu,
			 "ny" : self.ny,
			 "nx" : self.nx,
			 "ncolor" : self.ncolor}

		return p


	def dump(self, mode=0, subframe=None):
		''' 
		dump the light field data as one giant image
		in one of two formats: 
		0 - spatial (each subframe shows an individual camera capture)
		1 - angular (each subframe shows all of the light entering from a given direction)


		if subframe is None, then all subframes will be compiled into one giant image
		if subframe is a list, then dump will return a list where each element is the specified subframe

		'''

		pass


	def render(self, outputLF=False):
		
		## if we need to update the rendered image, then do so
		if self.changed:
			
			## get grid sampling points for the images
			UU,VV = np.mgrid[:self.nu, :self.nv]

			## set normalized weights
			w = self.mask / self.mask.sum()

			## clear any old renderings
			self.LFout = np.empty([self.ny,self.nx, self.nv, self.nu, self.ncolor], order="C", dtype=np.float32)

			I = np.empty([self.nv,self.nu, self.ncolor], order="C", dtype=np.float32)

			## cycle through the different camera positions and colors
			for y in xrange(self.ny):
				for x in xrange(self.nx):
					for c in xrange(self.ncolor):
						## map coordinates with linear spline interpolation
						## would sp.interpolate.RectBivariateSpline() be faster?
						I[:,:,c] = ndi.interpolation.map_coordinates(self.data, [UU + self.focus * (x - self.acx + 1),
																	  			 VV - self.focus * (y - self.acy + 1)],
																	  			 order=1, mode='constant')

					## copy resampled image to output LightField
					self.LFout[x,y] =  (I * w[:,:]).copy()
			## generate output image by adding the (weighted) resampled images
			self.output = self.LFout.sum((0,1))
			self.changed = False
		
		if outputLF:
			return self.output.copy(), self.LFOut.copy()
		else:
			return self.output.copy()


	def copy(self, newLF):
		''' copy all parameters and data to a new RLightField object '''

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
			newLF.focus= self.focus
			newLF.spacing = self.spacing
			newLF.output = np.empty_like(self.output)
			np.copyto(newLF.output, self.output)


	def set_data(self, data):
		''' set the lightfield data '''

		if data.ndim != 5:
			raise LFSizeError("Expected data to have 5 dimensions, instead found %d. set_data() aborted." % data.ndim)
		elif data.shape != self.data.shapes:
			raise LFShapeError("Expected data to have shape %s and instead found %s. set_data() aborted." % (str(self.data.shape), str(data.shape)))
		else:
			self.changed = True
			np.copyto(self.data, data)


	def refocus(self, foc):
		self.changed = True
		self.focus = foc

	def set_center(self, acx,acy):
		self.changed = True
		self.acx = acx
		self.acy = acy
		self.set_aperture(self.aperture)

	def set_aperture(self, ap):
		self.changed = True
		self.aperture = max(ap, 0)					## lower bound at zero
		self.aperture = min(ap, self.nx, self.ny)	## upper bound at edge length

		## set a circular aperture
		XX,YY = np.mgrid[:self.nx, :self.ny]
		x = XX - self.acx
		y = YY - self.acy

		mask = np.sqrt(x**2 + y**2) < self.aperture/2
		self.set_mask(mask)

	def set_mask(self, mask):
		self.changed = True
		np.copyto(self.mask, mask)

	def multiply_mask(self, mult):
		self.changed = True
		self.mask *= mult



# class ULightField(object):
# 	"""
# 	ULightField - Light Field from unstructured sampling points

# 	"""

# 	def __init__(self):
# 		pass

# 	def set_data(self):
# 		pass

# 	def get_data(self):
# 		pass

# 	def refocus(self):
# 		pass

# 	def set_DoF(self):
# 		pass

# 	def dump(self, mode=0):
# 		pass

# 	def render(self):
# 		pass

# 	def copy(self):
# 		pass
