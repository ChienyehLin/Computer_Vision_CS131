"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    for y in range(Hi):
        for x in range(Wi):
            out[y,x]=np.dot(padded[y:y+Hk,x:x+Wk].reshape(1,-1),kernel.reshape(-1,1))
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    pass
    k=(size-1)/2.0
    for y in range(size):
        for x in range(size):
            kernel[x,y]=1/(2.0*np.pi*sigma**2)*np.exp(((x-k)**2+(y-k)**2)/(-2*sigma**2))
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel=np.array([[-1,0,1]])/2
    out =    conv(img, kernel)
    ### END YOUR CODE
    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel=np.array([[-1],[0],[1]])/2
    out =    conv(img, kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx=partial_x(img)
    Gy=partial_y(img)
    G=np.sqrt(Gx**2+Gy**2)
    theta=(np.arctan(Gy/(Gx+0.000000000001))/np.pi*180)%360

    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    direction={0:(0,1),
               45:(1,1),
               90:(1,0),
               135:(1,-1),
               180:(0,-1),
               225:(-1,-1),
               270:(-1,0),
               315:(-1,1),
               360:(0,1)
              }

   

    for y in range(H-1):
        for x in range(W):
            y_neigh= y+direction[theta[y,x]][0]
            x_neigh= x+direction[theta[y,x]][1]
            y_neigh_prev=y-direction[theta[y,x]][0]
            x_neigh_prev=x-direction[theta[y,x]][1]
            
            
            alpha = np.deg2rad(theta[y, x])
            # note here the angle is measured clockwisely
            # i.e. if theta=90 degree the direction is south.
            p1=int(np.round(np.sin(alpha)))
            p2=int(np.round(np.cos(alpha)))
            #if(direction[theta[y,x]][0]!=p1 or direction[theta[y,x]][1]!=p2):
            #print(p1,direction[theta[y,x]][0],direction[theta[y,x]][1],p2,theta[y,x])
            if(y_neigh<0 or x_neigh<0):
                G_neigh =0
            elif(y_neigh>=H or x_neigh>=W):
                G_neigh =0
            else:
                 G_neigh = G[y_neigh,x_neigh]
            if(y_neigh_prev<0 or x_neigh_prev<0):
                G_neigh_prev =0
            elif(y_neigh_prev>=H or x_neigh_prev>=W):
                G_neigh_prev =0
            else:
                G_neigh_prev = G[y_neigh_prev,x_neigh_prev]

            if not (G[y, x] >= G_neigh_prev and G[y, x] >= G_neigh):
                out[y, x] = 0
            else:
                out[y, x] = G[y, x]
              
    ### END YOUR CODE
    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    H=img.shape[0]
    W=img.shape[1]
    strong_edges=  img>high
    weak_edges = (img < high) & (img > low)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)
    
    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    BFS_list=[]
    #1st iteration to get the weak_eadges connected to strong_edges
    for index in indices:
        neighbors=get_neighbors(index[0], index[1], H, W)
        for n_y,n_x in neighbors:
            if(weak_edges[n_y,n_x]==True):
                edges[n_y,n_x]=True      #make the weak_edge a strong_edge
                weak_edges[n_y,n_x]=False #delete correspending weak_edge
                BFS_list.append([n_y,n_x])
                
    #run BFS until no weak_edges connected to strong edges           
    while(len(BFS_list)>0):
        for index in BFS_list:
            neighbors=get_neighbors(index[0], index[1], H, W)
            for n_y,n_x in neighbors:
                if(weak_edges[n_y,n_x]==True):
                    edges[n_y,n_x]=True
                    weak_edges[n_y,n_x]=False
                    BFS_list.append([n_y,n_x])
            BFS_list.remove(index)
    #BFS
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel=gaussian_kernel(kernel_size, sigma)
    sobel=conv(img, kernel)
    G,theta=gradient(sobel)
    out=non_maximum_suppression(G,theta)
    strong,weak=double_thresholding(out, high, low)
    edge=link_edges(strong, weak)
    pass
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    H, W = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    pass
    for y,x in zip(ys,xs):
               for theta_index,theta in enumerate(thetas):
                        rho=cos_t[theta_index]*x+sin_t[theta_index]*y
                        rho_index=np.where(rhos==int(rho))
                        accumulator[rho_index,theta_index]+=1
    
    ### END YOUR CODE

    return accumulator, rhos, thetas
