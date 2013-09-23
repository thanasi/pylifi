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
# import pylab as pl


class RLightField(object):
	"""
	RectLightField - Light Field from samples on a rectangular grid

	necessary elements:
	resolution
	size
	colors
	mask
	data
	depth
	focus

	optional elements:
	spacing


	"""

	# maskstrings = ["all", "none", "x", "+", "=", "|", "||", "\\", "/", "v", "^", "-", "O", "o", "H"]


	def __init__(self, res=(1000,1000), arrsize=(5,5), colors=3, mask = None, depth=-1, foc=-1, spac=None):

		## image resolution
		self.resolution = res

		## array size
		self.size = arrsize

		## color depth (grayscale vs rgb)
		self.colors = colors

		## 2D mask for which camera views to use
		## default to all on
		if mask is None:
			self.mask = np.ones(arrsize, dtype=np.float32)

		if type(mask)==np.ndarray and mask.shape != arrsize:
			sys.stdout.write('(W) SGLightfield: Could not apply given mask to this lightfield. Reverting to default mask.\n')
			sys.stdout.flush()
			self.mask = np.ones(arrsize, dtype=np.float32)
		
		## set lightfield data
		## 5D array:
		## x_cam, y_cam, i, j, rgb/gray
		self.data = np.zeros(np.append(arrsize, [res, [colors]]), 
							 order="C",
							 dtype=np.float32)


		self.depth = depth
		self.set_depth(depth) ## set the depth of field
		self.focus = foc
		self.refocus(foc) 	  ## set the focus at infinity

		## [optional] grid spacing in (x,y) format
		self.spacing = spac


	def set_data(self, data):
		self.data = data

	def get_data(self):
		''' get the raw data '''
		return self.data

	def refocus(self, foc):
		pass

	def set_DoF(self, depth):
		pass

	def set_mask(self, mask):
		np.copyto(self.mask, mask)

	# def set_mask(self, pattern, rollx, rolly):
		# pass

	# def mult_mask(self, pattern, rollx, rolly):
		# pass

	def get_mask(self):
		return self.mask


	def dump(self, mode=0):
		''' 
		dump the light field data as one giant image
		in one of two formats: 
		0 - spatial (each subframe shows an individual camera capture)
		1 - angular (each subframe shows all of the light entering from a given direction)

		'''

		pass

	def render(self):
		output = np.zeros(np.append(self.resolution, [self.colors]),
						  dtype=np.float32, order="C")

		## fill the input
		
		return output

	def copy(self, newLF):
		''' copy all parameters and data to a new object '''

		newLF.resolution = self.resolution
		newLF.size = self.size
		newLF.colors = self.colors
		np.copyto(newLF.mask, self.mask)
		np.copyto(newLF.data, self.data)
		newLF.depth = self.depth
		newLF.focus= self.focus
		newLF.spacing = self.spacing





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
