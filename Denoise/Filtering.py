import numpy as np
import math
import cv2


class Filtering:

    def __init__(self, image, filter_name, filter_size, var=None):
        """initializes the variables of spatial filtering on an input image
        takes as input:
        image: the noisy input image
        filter_name: the name of the mask to use
        filter_size: integer value of the size of the mask
        alpha_d: parameter of the alpha trimmed mean filter
        order: parameter of the order for contra harmonic"""

        self.image = image

        if filter_name == 'arithmetic_mean':
            self.filter = self.get_arithmetic_mean
        elif filter_name == 'geometric_mean':
            self.filter = self.get_geometric_mean
        if filter_name == 'local_noise':
            self.filter = self.get_local_noise
        elif filter_name == 'median':
            self.filter = self.get_median
        elif filter_name == 'adaptive_median':
            self.filter = self.get_adaptive_median

        self.filter_size = filter_size
        self.global_var = var

        self.S_max = 15

    def get_arithmetic_mean(self, roi):
        """Computes the arithmetic mean filter
        takes as input:
        kernel: a list/array of intensity values
        returns the arithmetic mean value in the current kernel"""

        x = np.shape(roi)
        summation = 0
        for i in range(0, x[0]):
            summation = summation + roi[i]
        arithmetic = int(summation / len(roi));

        return arithmetic

    def get_geometric_mean(self, roi):
        """Computes the geometric mean filter
                        takes as input:
                        kernel: a list/array of intensity values
                        returns the geometric mean value in the current kernel"""

        x = np.shape(roi)
        mul = 1
        for i in range(0, x[0]):
            mul = mul * roi[i]
        geometric = int(math.pow(mul, 1 / len(roi)))
        return geometric

    def get_local_noise(self, kernel, roi):
        """Computes the result of local noise reduction
                        takes as input:
                        kernel: a list/array of intensity values
                        returns result of local noise reduction value of the current kernel"""

        arithematic_mean = self.get_arithmetic_mean(roi)
        zz = 0
        for i in range(0, len(roi)):
            zz = zz + (arithematic_mean - roi[i]) ** 2
        zz = zz / (len(roi))
        lclnoise = roi[(len(roi)) // 2] - ((self.global_var / zz) * (roi[(len(roi)) // 2] - arithematic_mean))
        return lclnoise

    def get_median(self, roi):
        """Computes the median filter
        takes as input:
        kernel: a list/array of intensity values
        returns the median value in the current kernel
        """

        med = np.median(roi)
        return med

    def get_adaptive_median(self, roi):
        """Computes the adaptive median filtering value
        Note: Adaptive median filter may involve additional steps, you are welcome to create any additional functions as needed,
        and you can change the signature of get_adaptive_median function as well.
                        takes as input:
        kernel: a list/array of intensity values
        returns the adaptive median filtering value"""

        roi = roi[:len(roi) - 2]
        if ((np.median(roi) - min(roi)) > 0) and ((np.median(roi) - max(roi)) < 0):
            if (((roi[int(len(roi) / 2)]) - min(roi)) > 0) and (((roi[int(len(roi) / 2)]) - max(roi)) < 0):
                return roi[int(len(roi) / 2)]
            else:
                return np.median(roi)
        else:
            self.filter_size += 2
            if self.filter_size <= self.S_max:
                r, c = self.image.shape
                ct = 0
                tmp_value = 0
                image_v = np.zeros([r + (self.filter_size - 1), c + (self.filter_size - 1)])
                image_r, image_c = image_v.shape

                for i in range(round((self.filter_size - 1) / 2), image_r):
                    for j in range(round((self.filter_size - 1) / 2), image_c):
                        if ct < r and tmp_value < c:
                            image_v[i, j] = self.image[ct, tmp_value]
                        tmp_value = tmp_value + 1
                    tmp_value = 0
                    ct = ct + 1
                i = int(roi[len(roi) - 2])
                j = int(roi[len(roi) - 1])
                roi = []
                for a in range(i - round((self.filter_size - 1) / 2), i + round((self.filter_size - 1) / 2) + 1):
                    for b in range(j - round((self.filter_size - 1) / 2), j + round((self.filter_size - 1) / 2) + 1):
                        roi.append(image_v[a, b])
                roi.append(i)
                roi.append(j)
                return self.get_adaptive_median(roi)
            else:
                return np.median(roi)

    def filtering(self):
        """performs filtering on an image containing gaussian or salt & pepper noise
        returns the denoised image
        ----------------------------------------------------------
        Note: Filtering for the purpose of image restoration does not involve convolution.
        For every pixel in the image, we select a neighborhood of values defined by the kernel and apply a mathematical
        operation for all the elements with in the kernel. For example, mean, median and etc.
        Steps:
        1. add the necesssary zero padding to the noisy image that way we have sufficient values to perform the operations
        on the border pixels. The number of rows and columns of zero padding is defined by the kernel size
        2. Iterate through the image and every pixel (i,j) gather the neighbors defined by the kernel into a list (or any data structure)
        3. Pass these values to one of the filters that will compute the necessary mathematical operations (mean, median, etc.)
        4. Save the results at (i,j) in the ouput image.
        5. return the output image

        Please note that the adaptive median filter may involve additional steps, you are welcome to create any additional functions as needed,
        and you can change the signature of get_adaptive_median function as well.
        """

        image_v = np.zeros(
            (self.image.shape[0] + 2 * (self.filter_size // 2), self.image.shape[1] + 2 * (self.filter_size // 2)))
        for a in range((self.filter_size // 2), self.image.shape[0] + (self.filter_size // 2)):
            for b in range((self.filter_size // 2), self.image.shape[1] + (self.filter_size // 2)):
                image_v[a, b] = self.image[a - (self.filter_size // 2), b - (self.filter_size // 2)]
        for a in range((self.filter_size // 2), self.image.shape[0] + (self.filter_size // 2)):
            for b in range((self.filter_size // 2), self.image.shape[1] + (self.filter_size // 2)):
                array_value = []
        for x in range(-(self.filter_size // 2), (self.filter_size // 2) + 1):
            for y in range(-(self.filter_size // 2), (self.filter_size // 2) + 1):
                array_value.append(image_v[a + x, b + y])
        array_value.append(a - (self.filter_size // 2))
        array_value.append(b - (self.filter_size // 2))
        filter_value = self.filter(array_value)
        self.image[a - (self.filter_size // 2), b - (self.filter_size // 2)] = filter_value
        return self.image
