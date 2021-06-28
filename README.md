# Pytorch Gradient Consistency Loss
A PyTorch implementation of the gradient consistency loss function, based on:

The gradient consistency loss is an implementation of the gradient correlation (GC), which is defined by the normalized cross correlation (NCC) between the gradients of two images. Given two images, A and B, the GC is defined as
	GC(A,B)=  1/2{NCC(∇_x A,∇_x B)+NCC(∇_y A,∇_y B)}
(3)
where ∇_x and ∇_y are the gradient operators in the horizontal and vertical directions, respectively. The NCC(A,B) is defined as 
	NCC(A,B)=  (∑_((i,j))▒〖(A-A ̅)(B-B ̅)〗)/(√(∑_((i,j))▒〖(A-A ̅)^2 〗) √(∑_((i,j))▒〖(B-B ̅)^2 〗))	(4)
where A ̅ and B ̅  represent the mean values of A and B, respectively. Using these equations, the gradient consistency loss L_GC can be defined as
	L_GC (G)= E_(x,y,z) [1-GC(y,G(x,z))]	(5)
where G is the generative model, x is the input image, z is a random noise vector, and y is the target image. By combing this loss function with the adversarial loss (equation 1) and L1 loss (equation 2), the complete objective function is defined as 
	G^*=arg⁡min┬G⁡max┬D⁡〖〖 L〗_adv 〗  +λ_L1  L_L1+λ_GC  L_GC	(6)
where λ_L1 and λ_GC are weights to balance the contribution of each loss.


Penney, G.P., et al.: A comparison of similarity measures for use in 2-D-3-D medical image registration. _IEEE transactions on medical imaging 17(4)_ (1998) 586–595 
