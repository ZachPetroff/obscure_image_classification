import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from scipy import signal
import skimage.measure
from tqdm import tqdm
from multiprocessing import Pool
import sys

def harris_detection(I, T):
    ''' Implements harris corner detection algo. '''
    # used to find partial derivatives
    sobel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    sobel_y = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    
    # used to calculate A, B, and C
    box = [[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1]]
    
    # convert to grayscale
    I = I.convert("L")
    
    # convert to NP array, so we can have values > 255
    im_array = np.array(I)

    # find partial derivatives
    Ix = signal.fftconvolve(im_array, sobel_x)
    Iy = signal.fftconvolve(im_array, sobel_y)

    # find A, B, and C
    A = Ix * Ix
    A = signal.fftconvolve(A, box)
    B = Ix * Iy
    B = signal.fftconvolve(B, box)
    C = Iy * Iy
    C = signal.fftconvolve(C, box)
    # Find eigenvalues
    trace = A + C
    det = A*C-B*B

    half_trace = trace / 2
    S = (trace*trace / 4 - det) ** .5
    
    lambda1 = half_trace + S
    lambda2 = half_trace - S 
    
    # find minimum eigenvalues
    min_eigens = element_oper(lambda1, lambda2, find_min)
    
    corner_mat = np.zeros((len(min_eigens),len(min_eigens[0])))
    # Apply threshold
    for i in range(len(corner_mat)):
        for j in range(len(corner_mat[0])):
            p = min_eigens[i,j]
            if p > T:
                corner_mat[i,j] = min_eigens[i,j] 

    return corner_mat

def element_oper(img1, img2, oper):
    ''' performs given element-by-element operation on 
        the two given images/matrices. '''
    if len(img1) != len(img2) or len(img1[0]) != len(img2[0]):
        print("Images must be the same size.")
    ret_mat = np.zeros((len(img1),len(img1[0])))
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            p1 = img1[i, j]
            p2 = img2[i, j]
            ret_mat[i, j] = oper(p1,p2)
    return ret_mat

def find_min(v1, v2):
    ''' returns min of v1 and v2. Used to find min eigen-values. '''
    return min(v1, v2)

def count_nonzero(corners):
  total = 0
  corners = corners.flatten()
  for i in range(len(corners)):
    if corners[i] != 0:
      total+=1
  return total / 82053

def get_corneravg(corners):
  corners = corners.flatten()
  return (sum(corners) / len(corners)) / 10738.403887336683

def get_perimeter(image):
  image = image.convert("L")
  image = np.array(image)
  return skimage.measure.perimeter_crofton(image) / 18816.16631495571

def get_partial_var(image):
  image = image.convert("L")
  image = np.array(image)
  px = signal.fftconvolve(image, [[-1,-1,-1], [0,0,0], [1,1,1]])
  py = signal.fftconvolve(image, [[-1,0,1], [-1,0,1], [-1,0,1]])
  p = px+py
  return np.var(p.flatten()) / 10575.527824097411


# Finds the maximum grayscale value in a convolution image given a percentage
#   - percentage must be input as a decimal, ie 10% = 0.1
def gmax(im,percentage):
    matrix = np.asarray(im) # convert to array
    window_width = int(im.width*percentage) # get window width
    window_height = int(im.height*percentage) # get window height
    gmax_val = -1
    for i in range(0, im.width-window_width, 3):
        # Known bug, included for time and consistency reasons
        #   - The following line should have range(im.height-window_height)
        #   - This version artificially increases the average grayscale values
        #     for convolutions whose average grayscale values near the bottom
        #     are negative
        for j in range(0, im.width-window_height, 3):  
            window = matrix[i:i+window_width,j:j+window_height]
            value = np.average(window)
            if value > gmax_val:
                gmax_val = value
    return gmax_val

# Mirrors pixels out of range for convolutions
def get_pixel(xypair,im):
    (x,y) = xypair
    if x<0:
        x = abs(x)
    elif x>=im.width:
        x = 2*im.width-x-2
    if y<0:
        y = abs(y)
    elif y>=im.height:
        y = 2*im.height-y-2    
    return im.getpixel((x,y))


# Main function
#   - Performs 6 convolutions at once
#   - Returns a numpy array containing the following variables (in this order)
#        lapneg_gavg - Average grayscale value after Laplacian negative filter
#        lappos_gavg - Average grayscale value after Laplacian positive filter
#        sobelh_gavg - Average grayscale value after Sobel Vertical filter
#        sobelv_gavg - Average grayscale value after Sobel Horizonatal filter
#        Horizontal Luminance   - Average across RGB of the average RGB tuple
#                                 after convolultion with luminance filter
#                                 (formlerly Sobel V Color Avg)
#        Vertical Edge Dilation - Average of a convolution that detects any
#                                 increase in grayscale value as you move up an
#                                 image & turns qualifying pixels to near-white
#                                 (formerly Sobel V Gray Avg)
#        Bluescale - Average value similar to grayscale calculated by 
#                    truncating the R and G components and amplifying the B
#                    (formerly Sobel H Color Avg)
#        sobelv_gmax50 - maximal grayscale value of Sobel V convolution across 
#                        all possible 25% area windows
#        sobelh_gmax50 - maximal grayscale value of Sobel H convolution across 
#                        all possible 25% area windows
#        sobelh_gmax10 - maximal grayscale value of Sobel H convolution across 
#                        all possible 1% area windows
#        lapneg_gmax50 - maximal grayscale value of Laplacian negative 
#                        convolution across all possible 25% area windows
#   - Returns a list containing the headers for the above variables
def get_features(im):
    image = Image.open(im)
    # Sobel vertical kernel
    sobelv = np.array([( 1/8, 2/8, 1/8),
                       ( 0,   0,   0  ),
                       (-1/8,-2/8,-1/8)])

    # Sobel horizontal kernel
    sobelh = np.array([(-1/8, 0, 1/8),
                       (-2/8, 0, 2/8),
                       (-1/8, 0, 1/8)])

    # Positive Laplacian
    lappos = np.array([(0, 1, 0),
                       (1, 4, 1),
                       (0, 1, 0)])
    # Negative Laplacian
    lapneg = np.array([( 0, -1,  0),
                       (-1,  4, -1),
                       ( 0, -1,  0)])

    # kernel list
    kernel = [sobelv, sobelh, lappos, lapneg]
    array_out = np.zeros(15)
    sobelv_g = Image.new("L",  (image.width, image.height), color=0)
    sobelh_g = Image.new("L",  (image.width, image.height), color=0)
    lappos_g = Image.new("L",  (image.width, image.height), color=0)
    lapneg_g = Image.new("L",  (image.width, image.height), color=0)
    sobelv_c = Image.new("RGB",(image.width, image.height), color=0)
    sobelh_c = Image.new("RGB",(image.width, image.height), color=0)
    sobelv_g2= Image.new("RGB",(image.width, image.height), color=0)
    im_g = image.convert("L")
    # Convolution step

    for x in range(1, image.width-1):
        for y in range(1, image.height-1):
            im_box = np.array([(image.getpixel((x-1,y-1)), image.getpixel((x,y-1)), image.getpixel((x+1,y-1))),
                                (image.getpixel((x-1,y  )), image.getpixel((x,y  )), image.getpixel((x+1,y  ))), 
                                (image.getpixel((x-1,y+1)), image.getpixel((x,y+1)), image.getpixel((x+1,y+1)))])

            im_g_box = np.array([(im_g.getpixel((x-1,y-1)), im_g.getpixel((x,y-1)), im_g.getpixel((x+1,y-1))),
                                  (im_g.getpixel((x-1,y  )), im_g.getpixel((x,y  )), im_g.getpixel((x+1,y  ))), 
                                  (im_g.getpixel((x-1,y+1)), im_g.getpixel((x,y+1)), im_g.getpixel((x+1,y+1)))])

            sobelv_g_pix = np.sum(im_g_box*kernel[0])
            sobelh_g_pix = np.sum(im_g_box*kernel[1])
            lappos_g_pix = np.sum(im_g_box*kernel[2])
            lapneg_g_pix = np.sum(im_g_box*kernel[3])
            sobelv_c_pix = np.sum(np.sum(im_box*kernel[0],axis=1),axis=0)
            sobelh_c_pix = np.sum(np.sum(im_box*kernel[1],axis=1),axis=0)
            
            sobelv_g.putpixel((x,y),int(sobelv_g_pix))
            sobelh_g.putpixel((x,y),int(sobelh_g_pix))
            lappos_g.putpixel((x,y),int(lappos_g_pix))
            lapneg_g.putpixel((x,y),int(lapneg_g_pix))
            sobelv_c.putpixel((x,y),(int(sobelv_c_pix[0]),int(sobelv_c_pix[1]),int(sobelv_c_pix[2])))            
            sobelh_c.putpixel((x,y),(int(sobelh_c_pix[0]),int(sobelh_c_pix[1]),int(sobelh_c_pix[2]))) 
            sobelv_g2.putpixel((x,y),int(sobelv_g_pix))

    # Variable Creation       
    #   - Laplacian negative average grayscale value (lapneg_gavg) 
    array_out[0] = np.average(np.array(lapneg_g)) / 4.782548828
    
    #   - Laplacian positive average grayscale value (lappos_gavg) 
    array_out[1] = np.average(np.array(lappos_g)) / 255
    
    #   - Sobel horizontal average grayscale value   (sobelh_gavg) 
    array_out[2] = np.average(np.array(sobelh_g)) / 2.397148438
    
    #   - Sobel vertical average grayscale value     (sobelv_gavg)
    array_out[3] = np.average(np.array(sobelv_g)) / 2.181048177
    
    #   - Partial Derivative Variance
    array_out[4] = get_partial_var(image)
    corners = harris_detection(image, 5000)
    
    #   - Number of Corners
    array_out[5] = count_nonzero(corners)
    
    #   - Average Corner Strength
    array_out[6] = get_corneravg(corners)

    #   - horizontal luminance change                (Sobel V Color Avg)
    array_out[7] = np.average(np.array(sobelv_c)) / 2.479327257
    
    #   - vertical edge dilation                     (Sobel V Gray Avg)
    array_out[8] = np.average(np.array(sobelv_g2)) / 89.23580187
    
    #   - bluescale                                 (Sobel H Color Avg)    
    array_out[9] = np.average(np.array(sobelh_c)) / 64.46461914
    
    #   - Sobel V maximal 25%-window grayscale Avg   (sobelv_gmax50)    
    array_out[10] = gmax(sobelv_g,.5) / 3.92578125
    
    #   - Sobel H maximal 25%-window grayscale Avg   (sobelv_gmax50)
    array_out[11] = gmax(sobelh_g,.5) / 5.617124782
    
    #   - Sobel H maximal  1%-window grayscale Avg   (sobelh_gmax10)
    array_out[12] = gmax(sobelh_g,.1) / 13.90416667

    #   - Lap. Neg. maximal 25%-window grayscale Avg (sobelv_gmax50)
    array_out[13]= gmax(lapneg_g,.5) / 8.107815735

    #   - Crofton Perimeter
    array_out[14] = get_perimeter(image)
    
    return array_out

def predict(feats):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='tanh', use_bias=False),
        tf.keras.layers.Dense(256, activation='tanh', use_bias=False),
        tf.keras.layers.Dense(256, activation='elu', use_bias=False),
        tf.keras.layers.Dense(256, activation='elu', use_bias=False),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])                                                                                                                                                                                                           
    
    model.load_weights("weights/model_weights.ckpt").expect_partial()
    feats = list(feats)
    del feats[-1]
    feats = np.array(feats)
    prediction = model.predict(np.array(feats).reshape(1, -1))[0][0]
    if prediction < .5:
        return 0
    else:
        return 1

if __name__ == '__main__':    
    image_path = sys.argv[1]
    feats = get_features(image_path)
    prediction = predict(feats)
    if prediction == 1:
        print('Prediction: Obscured')
    else:
        print('Prediction: Unobscured')