import cv2
import matplotlib.pyplot as plt

from compression import CompressionFactory, PSNR


def load_image(image_path: str, grayscale: bool = True, resize: tuple = None):
    """
    Load an image from a file.

    Args:
        image_path (str): Path to the image file.
        grayscale (bool): Whether to load the image in grayscale.
        resize (tuple): Resize dimensions (width, height).

    Returns:
        np.ndarray: The loaded image.
    """
    image = cv2.imread(
        image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    )
    if resize:
        image = cv2.resize(image, resize)
    return image


def plot_images(images, figsize=(18, 6)):
    """
    Plot a list of images with corresponding titles.

    Args:
        images (list of np.ndarray): Images to plot.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        ax = plt.subplot(1, len(images), i + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


psnr = PSNR()
image = load_image("./assets/lena.png", grayscale=True, resize=(512, 512))

# Apply DCT compression
dct_compression = CompressionFactory.create_compression("dct", block_size=8)
dct_compressed_image = dct_compression(image, quantization_factor=100)
psnr_dct = psnr(image, dct_compressed_image)

# Apply DWT compression
wavelet_compression = CompressionFactory.create_compression("haarwavelet")
dwt_compressed_image = wavelet_compression(image, level=2, quantization_factor=100)
psnr_dwt = psnr(image, dwt_compressed_image)

print(f"PSNR (DCT): {psnr_dct:.2f} dB")
print(f"PSNR (DWT): {psnr_dwt:.2f} dB")
images = [image, dct_compressed_image, dwt_compressed_image]
plot_images(images)
