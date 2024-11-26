import numpy as np
from abc import ABC, abstractmethod


class Compression(ABC):
    """
    Base class for compression algorithms.
    """

    def __init__(self, **kwargs):
        """
        Initialize the compression algorithm with any additional parameters.

        Args:
                **kwargs: Additional parameters specific to the compression algorithm.
        """
        pass

    @abstractmethod
    def compress(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compress an image using the specific algorithm.

        Args:
            image (np.ndarray): The input image.
            **kwargs: Additional parameters specific to the compression algorithm.

        Returns:
            np.ndarray: The compressed image.
        """
        pass

    @abstractmethod
    def decompress(self, data, **kwargs) -> np.ndarray:
        """
        Decompress the compressed data to reconstruct the image.

        Args:
            data: The compressed data.
            **kwargs: Additional parameters specific to the compression algorithm.

        Returns:
            np.ndarray: The reconstructed image.
        """
        pass

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compress and decompress the image in one step.

        Args:
            image (np.ndarray): The input image.
            **kwargs: Additional parameters for compression and decompression.

        Returns:
            np.ndarray: The reconstructed image after compression and decompression.
        """
        compressed = self.compress(image, **kwargs)
        reconstructed = self.decompress(compressed, **kwargs)
        return reconstructed
