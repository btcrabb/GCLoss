# Pytorch Gradient Consistency Loss
A PyTorch implementation of the gradient consistency loss function, based on:

Penney, G.P., et al.: A comparison of similarity measures for use in 2-D-3-D medical image registration. _IEEE transactions on medical imaging 17(4)_ (1998) 586–595 

and 

Hiasa Y, Otake Y, Takao M, et al. Cross-modality image synthesis from unpaired data using CycleGAN. _Springer_; 2018:31-41.


****************************************************************************************************************************

The objective function of a conditional GAN has been previously well defined and can be expressed as,

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bequation%7D%20L_%7Badv%7D%28G%2CD%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%2Cy%7D%5BlogD%28x%2C%20y%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%2Cz%7D%5Blog%281%20-%20D%28x%2C%20G%28x%2C%20z%29%29%5D%2C%20%5Cend%7Bequation%7D)

where the discriminator, D, tries to maximize the objective and the generator, G, tries to minimize it.\textsuperscript{9} This term is described as the adversarial loss; however, it is also paired with a more traditional L1 distance loss function defined as,

\begin{table}[H]
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{0.39cm}p{15.31cm}p{0.81cm}}
\multicolumn{1}{p{0.39cm}}{} & 
\multicolumn{1}{p{15.31cm}}{\begin{equation}
L_{L1}(G)  = \mathbb{E}_{x,y,z} [\left\Vert y - G(x,z)\right\Vert_{1}].
\end{equation}
} & 
\multicolumn{1}{p{0.81cm}}{(2)} \\ 
\end{tabular}
\end{adjustbox}
\end{table}
\vspace{1\baselineskip}
\textbf{\textit{Gradient consistency loss}}

To improve the anatomical accuracy of the resultant images by emphasizing the borders of the vasculature, equations (1) and (2) were also paired with a gradient consistency loss. The gradient consistency loss is an implementation of the gradient correlation (GC), which is defined by the normalized cross correlation (NCC) between the gradients of two images.\textsuperscript{7, 10} Given two images, A and B, the GC is defined as

\begin{table}[H]
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{0.39cm}p{15.31cm}p{0.81cm}}
\multicolumn{1}{p{0.39cm}}{} & 
\multicolumn{1}{p{15.31cm}}{\begin{equation}
GC\left(A,B\right) = \frac{1}{2}\{ NCC\left(\triangledown_{x}A, \triangledown_{x}B\right) + NCC\left(\triangledown_{y}A, \triangledown_{y}B\right)\} 
\end{equation}
} & 
\multicolumn{1}{p{0.81cm}}{(3)} \\ 
\end{tabular}
\end{adjustbox}
\end{table}
\vspace{1\baselineskip}
where \( \triangledown_{x}\) and \( \triangledown_{y}\) are the gradient operators in the horizontal and vertical directions, respectively. The \( NCC(A,B)\) is defined as 

\begin{table}[H]
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{0.39cm}p{15.31cm}p{0.81cm}}
\multicolumn{1}{p{0.39cm}}{} & 
\multicolumn{1}{p{15.31cm}}{\begin{equation}
NCC\left(A,B\right) = \frac{\sum_{(i,j)}^{}(A -\overline{A})(B -\overline{B})}{\sqrt{\sum_{(i,j)}^{}(A -\overline{A})^{2}}\sqrt{\sum_{(i,j)}^{}(B -\overline{B})^{2}}}
\end{equation}
} & 
\multicolumn{1}{p{0.81cm}}{(4)} \\ 
\end{tabular}
\end{adjustbox}
\end{table}
\vspace{1\baselineskip}
where \(\overline{A}\) and \(\overline{B}\)represent the mean values of \( A\) and \( B\), respectively. Using these equations, the gradient consistency loss \( L_{GC}\) can be defined as

\begin{table}[H]
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{0.39cm}p{15.31cm}p{0.81cm}}
\multicolumn{1}{p{0.39cm}}{} & 
\multicolumn{1}{p{15.31cm}}{\begin{equation}
L_{GC}\left(G\right) = \mathbb{E}_{x,y,z}[1 - GC\left(y,G\left(x,z\right)\right)]
\end{equation}
} & 
\multicolumn{1}{p{0.81cm}}{(5)} \\ 
\end{tabular}
\end{adjustbox}
\end{table}
\vspace{1\baselineskip}
where \( G\) is the generative model, \( x\) is the input image, \( z\) is a random noise vector, and \( y\) is the target image. By combing this loss function with the adversarial loss (1) and L1 loss (2), the complete objective function is defined as 

\begin{table}[H]
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{0.39cm}p{15.31cm}p{0.81cm}}
\multicolumn{1}{p{0.39cm}}{} & 
\multicolumn{1}{p{15.31cm}}{\begin{equation}
G^{\ast } = arg\min_{G}\max_{D} L_{adv} + \lambda_{L1} L_{L1} + \lambda_{GC} L_{GC}
\end{equation}
} & 
\multicolumn{1}{p{0.81cm}}{(6)} \\ 
\end{tabular}
\end{adjustbox}
\end{table}
\vspace{1\baselineskip}
where \( \lambda_{L1}\) and \( \lambda_{GC}\) are weights to balance the contribution of each loss.

\end{document}
