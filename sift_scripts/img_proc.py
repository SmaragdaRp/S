import cv2
import numpy as np
import os

# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html


def image_filtering(img):
    kernel = np.ones((5, 5), np.float32)/25
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def image_blurring(img):
    blur = cv2.blur(img, (5, 5))
    return blur


def gaussian_blurring(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return blur


def median_blurring(img):
    median = cv2.medianBlur(img, 5)
    return median


def bilateral_filtering(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def process_img(images, labels, train_label):

    temp = []
    pos = 0
    directory = "transformed_images"

    for i in range(len(images)):

        if labels[i] == train_label:
            pos = i
            break

    # 0. Add the initial photo first
    temp.append(images[pos])

    # 1. Filtered
    temp_img = image_filtering(images[pos])
    temp.append(temp_img)
    filename = "filter.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 2. Double filtered
    temp_img = image_filtering(temp_img)
    temp.append(temp_img)
    filename = "d_filter.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 3. Blurred
    temp_img = image_blurring(images[pos])
    temp.append(temp_img)
    filename = "blur.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 4. Double blurred
    temp_img = image_blurring(temp_img)
    temp.append(temp_img)
    filename = "d_blur.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 5. Gaussian blurred
    temp_img = gaussian_blurring(images[pos])
    temp.append(temp_img)
    filename = "gauss_blur.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 6. Double gaussian blurred
    temp_img = gaussian_blurring(temp_img)
    temp.append(temp_img)
    filename = "d_gauss_blur.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 7. Median blurred
    temp_img = median_blurring(images[pos])
    temp.append(temp_img)
    filename = "median_filter.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 8. Double median blurred
    temp_img = median_blurring(temp_img)
    temp.append(temp_img)
    filename = "d_median_filter.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 9. Bilateral filtered
    temp_img = bilateral_filtering(images[pos])
    temp.append(temp_img)
    filename = "bilateral.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    # 10. Double bilateral filtered
    temp_img = bilateral_filtering(temp_img)
    temp.append(temp_img)
    filename = "d_bilateral.jpg"
    cv2.imwrite(os.path.join(directory, filename), temp[-1])

    return temp
