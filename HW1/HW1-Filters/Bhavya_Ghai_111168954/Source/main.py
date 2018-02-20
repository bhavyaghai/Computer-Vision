# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending
# RUN Commands
# Q1 python main.py 1 input1.jpg ./
# Q2 python main.py 2 input2.png blurred2.exr ./
# Q1 python main.py 3 input3A.jpg input3B.jpg ./
# yapf -i --style="{based_on_style: google, indent_width: 4}" main.py

import os
import sys
import cv2
import numpy


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Write histogram equalization here
    img_out = img_in  # Histogram eqalization result

    # Split image into 3 colors
    b, g, r = cv2.split(img_in)

    # Calculate frequency for each pixel value for each color
    hist_b, bins = numpy.histogram(b.flatten(), 256, [0, 256])
    hist_g, bins = numpy.histogram(g.flatten(), 256, [0, 256])
    hist_r, bins = numpy.histogram(r.flatten(), 256, [0, 256])

    # Calculate Cumulative frequency for each pixel value for each color
    cdf_b = numpy.cumsum(hist_b)
    cdf_g = numpy.cumsum(hist_g)
    cdf_r = numpy.cumsum(hist_r)

    cdf_m_b = numpy.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_b - cdf_b.min()) * 255 / (cdf_b.max() - cdf_b.min())
    cdf_b = numpy.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = numpy.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_g - cdf_g.min()) * 255 / (cdf_g.max() - cdf_g.min())
    cdf_g = numpy.ma.filled(cdf_m_g, 0).astype('uint8')

    cdf_m_r = numpy.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_r - cdf_r.min()) * 255 / (cdf_r.max() - cdf_r.min())
    cdf_r = numpy.ma.filled(cdf_m_r, 0).astype('uint8')

    img_out = cv2.merge((cdf_b[b], cdf_g[g], cdf_r[r]))

    # VAlidation
    #equ = cv2.merge((cv2.equalizeHist(b), cv2.equalizeHist(g),
    #                 cv2.equalizeHist(r)))
    #res = numpy.hstack((img_out, equ))  #stacking images side-by-side
    #cv2.imwrite('res.png', res)

    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def high_pass_filter(img_in):

    # Write low pass filter here
    img_out = img_in  # Low pass filter result
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    f = numpy.fft.fft2(img)
    fshift = numpy.fft.fftshift(f)
    magnitude_spectrum = numpy.log(numpy.abs(fshift))
    magnitude_spectrum = numpy.array(magnitude_spectrum * 15, dtype=numpy.uint8)

    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows / 2, cols / 2
    fshift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_back = numpy.fft.ifft2(f_ishift)
    img_back = numpy.abs(img_back)

    img_out = img_back
    return True, img_out


def low_pass_filter(img_in):

    # Write high pass filter here
    img_out = img_in  # High pass filter result

    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    f = numpy.fft.fft2(img)
    fshift = numpy.fft.fftshift(f)
    magnitude_spectrum = numpy.log(numpy.abs(fshift))
    magnitude_spectrum = numpy.array(magnitude_spectrum * 15, dtype=numpy.uint8)

    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows / 2, cols / 2
    fshift[0:crow - 10, :] = 0
    fshift[crow + 10:, :] = 0
    fshift[:, 0:ccol - 10] = 0
    fshift[:, ccol + 10:] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_back = numpy.fft.ifft2(f_ishift)
    img_back = numpy.abs(img_back)
    img_out = img_back
    return True, img_out


def deconvolution(img_in):

    # Write deconvolution codes here
    img_out = img_in  # Deconvolution result
    img = img_in

    def ft(im, newsize=None):
        dft = numpy.fft.fft2(numpy.float32(im), newsize)
        return numpy.fft.fftshift(dft)

    def ift(shift):
        f_ishift = numpy.fft.ifftshift(shift)
        img_back = numpy.fft.ifft2(f_ishift)
        return numpy.abs(img_back)

    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    #De-covolution
    imf = ft(img, (img.shape[0], img.shape[1]))  # make sure sizes match
    gkf = ft(gk, (img.shape[0], img.shape[1]))  # so we can multiple easily
    imconvf = imf / gkf

    # now for example we can reconstruct the blurred image from its FT
    img_out = ift(imconvf)

    img_out = numpy.array(img_out, dtype=numpy.float32) * 255
    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    # Write laplacian pyramid blending codes here
    img_out = img_in1  # Blending result

    A, B = img_in1, img_in2

    # make images rectangular
    A = A[:, :A.shape[0]]
    B = B[:A.shape[0], :A.shape[0]]
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = numpy.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

# now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    img_out = ls_
    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
