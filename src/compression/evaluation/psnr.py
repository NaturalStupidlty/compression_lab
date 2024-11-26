import numpy as np


class PSNR:
    """
    Peak Signal-to-Noise Ratio (PSNR) is a metric used to measure the quality of an image after compression.

    PSNR is defined as:
    PSNR = 20 * log10(MAX_pixel / sqrt(MSE))

    where:
    - MAX_pixel: maximum pixel value of the image (e.g., 255 for 8-bit grayscale images)
    - MSE: Mean Squared Error between the original and compressed images

    https://uk.wikipedia.org/wiki/PSNR
    """

    def __call__(
        self, original: np.ndarray, compressed: np.ndarray, max_pixel: float = 255.0
    ):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return 100

        return 20 * np.log10(max_pixel / np.sqrt(mse))
