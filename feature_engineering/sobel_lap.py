import numpy as np
from PIL import Image

# Finds the maximum grayscale value in a convolution image given a percentage
#   - percentage must be input as a decimal, ie 10% = 0.1
def gmax(im,percentage):
    matrix = np.asarray(im)
    window_width = int(im.width*percentage)
    window_height = int(im.height*percentage)
    gmax_val = -1
    for i in range(im.width-window_width):
        # Known bug, included for time and consistency reasons
        #   - The following line should have range(im.height-window_height)
        #   - This version artificially increases the average grayscale values
        #     for convolutions whose average grayscale values near the bottom
        #     are negative
        for j in range(im.width-window_height):  
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
def sobel_lap(im):
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
    headers = ["lapneg_gavg", "lappos_gavg", "sobelh_gavg", "sobelv_gavg",
               "Horizontal Luminance", "Vertical Edge Dilation", "Bluescale",
               "sobelv_gmax50", "sobelh_gmax50", "sobelh_gmax10", "lapneg_gmax50"]
    array_out = np.zeros(11)
    sobelv_g = Image.new("L",  (im.width, im.height), color=0)
    sobelh_g = Image.new("L",  (im.width, im.height), color=0)
    lappos_g = Image.new("L",  (im.width, im.height), color=0)
    lapneg_g = Image.new("L",  (im.width, im.height), color=0)
    sobelv_c = Image.new("RGB",(im.width, im.height), color=0)
    sobelh_c = Image.new("RGB",(im.width, im.height), color=0)
    sobelv_g2= Image.new("RGB",(im.width, im.height), color=0)
    im_g = im.convert("L")
    # Convolution step
    for x in range(im.width):
        for y in range(im.height):
            try:
                im_box = np.array([(im.getpixel((x-1,y-1)), im.getpixel((x,y-1)), im.getpixel((x+1,y-1))),
                                   (im.getpixel((x-1,y  )), im.getpixel((x,y  )), im.getpixel((x+1,y  ))), 
                                   (im.getpixel((x-1,y+1)), im.getpixel((x,y+1)), im.getpixel((x+1,y+1)))])

                im_g_box = np.array([(im_g.getpixel((x-1,y-1)), im_g.getpixel((x,y-1)), im_g.getpixel((x+1,y-1))),
                                     (im_g.getpixel((x-1,y  )), im_g.getpixel((x,y  )), im_g.getpixel((x+1,y  ))), 
                                     (im_g.getpixel((x-1,y+1)), im_g.getpixel((x,y+1)), im_g.getpixel((x+1,y+1)))])

            except:
                im_box = np.array([(get_pixel((x-1,y-1),im), get_pixel((x,y-1),im), get_pixel((x+1,y-1),im)),
                                   (get_pixel((x-1,y  ),im), get_pixel((x,y  ),im), get_pixel((x+1,y  ),im)), 
                                   (get_pixel((x-1,y+1),im), get_pixel((x,y+1),im), get_pixel((x+1,y+1),im))])

                im_g_box = np.array([(get_pixel((x-1,y-1),im_g), get_pixel((x,y-1),im_g), get_pixel((x+1,y-1),im_g)),
                                     (get_pixel((x-1,y  ),im_g), get_pixel((x,y  ),im_g), get_pixel((x+1,y  ),im_g)), 
                                     (get_pixel((x-1,y+1),im_g), get_pixel((x,y+1),im_g), get_pixel((x+1,y+1),im_g))])

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
    array_out[0] = np.average(np.array(lapneg_g))
    
    #   - Laplacian positive average grayscale value (lappos_gavg) 
    array_out[1] = np.average(np.array(lappos_g))
    
    #   - Sobel horizontal average grayscale value   (sobelh_gavg) 
    array_out[2] = np.average(np.array(sobelh_g))
    
    #   - Sobel vertical average grayscale value     (sobelv_gavg)
    array_out[3] = np.average(np.array(sobelv_g))
    
    #   - horizontal luminance change                (Sobel V Color Avg)
    array_out[4] = np.average(np.array(sobelv_c))
    
    #   - vertical edge dilation                     (Sobel V Gray Avg)
    array_out[5] = np.average(np.array(sobelv_g2))
    
    #   - bluescale                                 (Sobel H Color Avg)    
    array_out[6] = np.average(np.array(sobelh_c))
    
    #   - Sobel V maximal 25%-window grayscale Avg   (sobelv_gmax50)    
    array_out[7] = gmax(sobelv_g,.5)
    
    #   - Sobel H maximal 25%-window grayscale Avg   (sobelv_gmax50)
    array_out[8] = gmax(sobelh_g,.5)
    
    #   - Sobel H maximal  1%-window grayscale Avg   (sobelh_gmax10)
    array_out[9] = gmax(sobelh_g,.1)

    #   - Lap. Neg. maximal 25%-window grayscale Avg (sobelv_gmax50)
    array_out[10]= gmax(lapneg_g,.5)
    
    # Normalization step
    max_vals = np.array([4.782548828, 255,         2.397148438, 2.181048177,
                         2.479327257, 89.23580187, 64.46461914,	3.92578125,
                         5.617124782, 13.90416667, 8.107815735])
    array_out /= max_vals

    return array_out, headers
