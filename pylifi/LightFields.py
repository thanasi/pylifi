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
import numpy as np
# import pylab as pl


class SGLightField(object):
	"""
	SGLightField - Light Field from samples on a structured grid

	"""

	def __init__(self, res=(1000,1000), arrsize=(5,5), mono=True, mask = None):
		self.resolution = res
		self.size = arrsize
		self.monochrome = True

		if mask is None or mask.shape != arrsize:
			self.mask = np.ones(arrsize)

		
		self.data = np.zeros()


	def set_data(self, data):
		self.data = data

	def get_data(self):
		return self.data

	def refocus(self, depth):
		pass

	def set_DoF(self, ):
		pass

	def dump(self, mode=0):
		pass

	def render(self):
		pass

	def copy(self):
		pass


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
