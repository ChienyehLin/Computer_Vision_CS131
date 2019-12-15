import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = io.imread(image_path)
    print(out.dtype)
    ### YOUR CODE HERE
    # Use skimage io.imread
    pass
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64)/255 #range of pixel values changed to 0.0 - 1.0 from 0-255
    print(out.dtype)
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = np.empty_like(image)#creat m*n matrix like image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for i in range(len(image[x][y])):### calculate new values in different RGB channel
                out[x][y][i]= 0.5*image[x][y][i]**2

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = color.rgb2gray(image)

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
   
    out = image.copy()
    if(channel=='R'):
        out[:,:,0]=0
    elif(channel =='G'):
        out[:,:,1]=0
    elif(channel == 'B'):
        out[:,:,2]=0   

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = lab.copy()
    out=(lab + [0, 128, 128]) / [100, 255, 255]# LAB range L: 0-100 A:-128+127 B: -128+127
    if(channel=='L'):
        out=out[:,:,0]
    elif(channel =='A'):
        out[:,:2]=0
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                for i in range(out.shape[2]):
                    out[x][y][0]=out[x][y][1]
                    out[x][y][1]=1-out[x][y][1]
    elif(channel == 'B'):
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                for i in range(out.shape[2]):
                    out[x][y][0]=out[x][y][2] 
                    out[x][y][1]=out[x][y][2] 
                    out[x][y][2]=1-out[x][y][2]

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = hsv
    channel_dict={'H':0,'S':1,'V':2}
    
    out =hsv[:,:,channel_dict[channel]]

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    left =  rgb_exclusion(image1,'R')
    right =  rgb_exclusion(image2,'G')
    left = left[:,:151,:]
    right= right[:,151:,:]
    out=np.concatenate((left,right),axis=1)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    length= round(image.shape[0]/2)#half length
    width= round(image.shape[1]/2)#half width
    out = None
    Top_left=rgb_exclusion(image[:length,:width,:],'R')
    Top_right=dim_image(image[:length,width:,:])
    Bottom_left= (image[:,:,:]**0.5)[length:,:width,:]
    Bottom_right=rgb_exclusion(image[length:,width:,:],'R')
    
    Top_half=np.concatenate((Top_left,Top_right),axis=1)
    Bottom_half=np.concatenate((Bottom_left,Bottom_right),axis=1)
   
    out=np.concatenate((Top_half,Bottom_half),axis=0)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
