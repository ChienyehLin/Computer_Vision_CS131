"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    

    ### YOUR CODE HERE
    flip_kernel=np.zeros((Hk, Wk))
    for i in range(Hk):
        for s in range(Wk):
            flip_kernel[i,s]=kernel[Hk-1-i,Wk-1-s]
            
    for i in range(Hi):
        for s in range(Wi):
            conv_value=0
            for y in range(Hk):
                for x in range(Wk):
                    if(i-1+y<0 or s-1+x<0 or i-1+y>=Hi or s-1+x>=Wi):
                        image_value=0
                    else:
                        image_value=image[i-1+y,s-1+x]
                    conv_value+=image_value*flip_kernel[y,x]
            out[i,s]=conv_value
                    
    ''' for i in range(Hi-Hk+1):
        for s in range(Wi-Wk+1):
            out[i+Hk//2,s+Wk//2]=np.sum(np.multiply(flip_kernel,image[i:i+Hk,s:s+Wk]))
         
    '''        
    pass
    ### END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out=np.zeros((H+2*pad_height,W+2*pad_width))
    for i in range(H):
        for s in range(W):
            out[i+pad_height,s+pad_width]=image[i,s]
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    new_kernel=np.flip(np.flip(kernel, 0),1)

    pad_width,pad_height= Wk//2,Hk//2
    padded=zero_pad(image,pad_height,pad_width)
    ### YOUR CODE HERE
    for i in range (Hi):
        for s in range(Wi):
            out[i,s]= np.sum(np.multiply(padded[i:i+Hk,s:s+Wk],new_kernel))
            
    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    new_kernel=np.flip(np.flip(kernel, 0),1)

    pad_width,pad_height= Wk//2,Hk//2
    padded=zero_pad(image,pad_height,pad_width)
    ### YOUR CODE HERE
    for i in range (Hi):
        for s in range(Wi):
            out[i,s]= np.dot(padded[i:i+Hk,s:s+Wk].reshape(1,Hk*Wk),new_kernel.reshape( Hk*Wk,1))
            
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    ### YOUR CODE HERE
    new_kernel=np.flip(np.flip(g, 0),1)
    out=conv_fast(f,new_kernel)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
   
    ### YOUR CODE HERE
    new_kernel=np.flip(np.flip(g, 0),1)
    new_kernel=new_kernel-np.sum(new_kernel)/(g.shape[0]*g.shape[1])
    out=conv_fast(f,new_kernel)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    pad_width,pad_height= Wk//2,Hk//2
    new_kernel=(g-np.mean(g))/np.std(g)
    padded=zero_pad(f,pad_height,pad_width)
    
    for i in range (Hi):
        for s in range(Wi):
            patch_image= padded[i:i+Hk,s:s+Wk].reshape(1,Hk*Wk)
            out[i,s]= np.dot((patch_image-np.mean(patch_image))/np.std(patch_image),new_kernel.reshape( Hk*Wk,1))
            
    pass
    ### END YOUR CODE

    return out
