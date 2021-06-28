import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

def gradient(x):
	# Calculates the gradients in the x and y directions for an array
	# idea from tf.image.image_gradients(image)
	# https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
	# x: (b,c,h,w), float32 or float64
	# returns: dx, dy: (b,c,h,w)

	h_x = x.size()[-2]
	w_x = x.size()[-1]
	# gradient step=1
	left = x

	right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]

	top = x
	bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

	dx, dy = torch.sub(right,left), torch.sub(bottom,top) 
	dx[:, :, :, -1] = 0
	dy[:, :, -1, :] = 0

	return dx, dy

def ncc(A, B):
	# Calculates the normalized cross correlation between array A and B
	# A and B: (b,c,h,w), float32 or float64
	# returns: float

	numerator = torch.sum(torch.mul(torch.sub(A, torch.mean(A)),torch.sub(B, torch.mean(B))))

	denominatorA = torch.sqrt(torch.sum(torch.sub(A,torch.mean(A))**2))
	denominatorB = torch.sqrt(torch.sum(torch.sub(B,torch.mean(B))**2))

	return torch.div(numerator, torch.mul(denominatorA, denominatorB))


def gc(tensorA, tensorB):
	# Calculates the gradient correlation between tensor A and B
	# A and B: (b,c,h,w), float32 or float64
	# returns: float

	dxA, dyA = gradient(tensorA)
	dxB, dyB = gradient(tensorB)

	loss = torch.mul(0.5, torch.add(ncc(dxA, dxB), ncc(dyA,dyB)))

	return torch.sub(1,loss)
	
def gc_with_gradient(tensorA, tensorB):
	# implentation of GC function above with image gradients.
	# A and B: (b,c,h,w), float32 or float64
	# returns: float, array, array
	
	A = Variable(tensorA, requires_grad=True)
	B = Variable(tensorB, requires_grad=True)
	
	loss = gc(A, B)
	loss.backwards()
	gradA = A.grad
	gradB = B.grad
	
	return loss, gradA, gradB
	
