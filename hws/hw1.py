import math
import glob

import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import convolve2d


HOUGH_SPACE_SIZE = 512
LINE_MIN_LENGTH = 64
NON_MAX_SUPPRESSION_DIST = 5

# TODO: pick the required threshold
LINE_EDGE_THRESHOLD = 250


def imshow_with_colorbar(index, img):
    plt.figure(index)
    ax = plt.subplot(111)
    im = ax.imshow(img)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def convolve(img, kernel):
    # TODO:
    # write convolve operation
    # (the output array may have negative values)
    # for result comparison:
    # return convolve2d(img, kernel, mode='valid')

    img_m, img_n  = img.shape
    kernel_m, kernel_n = kernel.shape

    convolved_m = img_m - kernel_m + 1
    convolved_n = img_n - kernel_n + 1

    convolved_img = np.zeros((convolved_m, convolved_n))

    for i in range(convolved_m * convolved_n):
        y = i // convolved_m
        x = i % convolved_n

        convolved_img[y, x] = (img[y:y+kernel_m, x:x+kernel_n] * kernel).sum()

    return convolved_img


def threshold(img, threshold_value):
    # TODO:
    # write threshold operation
    # funciton should return numpy array with 1 in pixels that are higher or equal to threshold, otherwise - zero

    return (img >= threshold_value).astype(np.uint8)


def draw_line_in_hough_space(h_space, y, x):
    # TODO:
    # I propose to use polar coordinates instead of proposed y = a*x + b.
    # They are way easier for implementation.
    # The idea is the same. Short documentation: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

    for angle_index in range(1, HOUGH_SPACE_SIZE):
        angle = angle_index / HOUGH_SPACE_SIZE * (2 * math.pi)
        r_index = math.floor(x * math.cos(angle) + y * math.sin(angle))

        if r_index <= 0:
            continue

        h_space[r_index, angle_index] += 1


def count_lines(img, is_debug):
    if is_debug: imshow_with_colorbar(1, img)

    # TODO:
    # pick the kernel for convolve operation, you need to find edges
    k = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])

    # NOTE: Uncomment to use kernel below
    # k = np.array([
    #     [0, -1, 0],
    #     [-1, 4, -1],
    #     [0, -1, 0]
    # ])

    # calculate convolve operation
    img_c = convolve(img, k)
    assert (img_c - convolve2d(img, k, mode='valid')).sum() == 0, "Incorrect convolution operation is implementation! "

    if is_debug: imshow_with_colorbar(2, img_c)

    # TODO:
    # apply thresholding (the result of the threshold should be array with zeros and ones)
    img_thr = threshold(img_c, LINE_EDGE_THRESHOLD)  # LINE_EDGE_THRESHOLD
    if is_debug: imshow_with_colorbar(3, img_thr)

    h_space = np.zeros((HOUGH_SPACE_SIZE, HOUGH_SPACE_SIZE), dtype=np.int)

    # for each coordinate ...
    for y in range(img_thr.shape[0]):
        for x in range(img_thr.shape[1]):
            # if there is edge ...
            if img_thr[y,x] != 0:
                draw_line_in_hough_space(h_space, y, x)
    if is_debug: imshow_with_colorbar(4, h_space)

    # apply threshold for hough space.
    # TODO:
    # pick the threshold to cut-off smaller lines
    # h_space_mask = threshold(h_space, ...)
    h_space_mask = threshold(h_space, LINE_MIN_LENGTH)
    if is_debug: imshow_with_colorbar(5, h_space_mask)

    # get indices of non-zero elements
    y_arr, x_arr = np.nonzero(h_space_mask)

    # calculate the lines number,
    # here is simple non maximum suppresion algorithm.
    # it counts only one point in the distance NON_MAX_SUPPRESSION_DIST
    lines_num = 0
    for i in range(len(y_arr)):
        has_neighbour = False
        for j in range(i):
            yi, xi = y_arr[i], x_arr[i]
            yj, xj = y_arr[j], x_arr[j]
            dy = abs(yi - yj)
            dx = abs(xi - xj)
            if dy <= NON_MAX_SUPPRESSION_DIST and \
                (dx <= NON_MAX_SUPPRESSION_DIST or
                 dx >= HOUGH_SPACE_SIZE - NON_MAX_SUPPRESSION_DIST):  # if x axis represents the angle, than check the distance if the points points that are near 0 degree (for example, distance between 1 deg. and 359 deg. is 2 deg.)
                has_neighbour = True
                break

        if not has_neighbour:
            lines_num += 1

    if is_debug: print('lines number: %d' % lines_num); plt.show()
    return lines_num


if __name__ == '__main__':
    fpath_arr = glob.glob('../images/hw1/*.png')
    fpath_arr.sort()

    print(fpath_arr)

    for img_fpath in fpath_arr:
        img = cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE)
        print('Number of lines: %d' % count_lines(img, True))
