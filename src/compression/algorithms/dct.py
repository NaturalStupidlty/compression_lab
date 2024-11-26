import numpy as np

from compression import Compression


class DCTCompression(Compression):
    def __init__(self, block_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.__image_shape = None
        self.__block_size = block_size
        self.__transform_matrix = self.__compute_transform_matrix(block_size)
        self.__transform_matrix_transposed = self.__transform_matrix.T

    @staticmethod
    def __compute_transform_matrix(block_size: int):
        alpha = np.ones(block_size)
        alpha[0] = 1 / np.sqrt(2)
        alpha = alpha * np.sqrt(2 / block_size)

        u = np.arange(block_size).reshape(-1, 1)
        x = np.arange(block_size).reshape(1, -1)

        return alpha.reshape(-1, 1) * np.cos(
            ((2 * x + 1) * u * np.pi) / (2 * block_size)
        )

    def compress(
        self, image: np.ndarray, quantization_factor: int = 100, **kwargs
    ) -> list:
        self.__image_shape = image.shape
        blocks = self.__divide_into_blocks(image)
        blocks = self.__dct(blocks)
        return self.quantize_blocks(blocks, quantization_factor)

    def decompress(
        self, blocks: list, image_shape: tuple = None, **kwargs
    ) -> np.ndarray:
        blocks = self.__inverse_dct(blocks)
        image_shape = image_shape or self.__image_shape
        return self.reconstruct_image(blocks, image_shape, self.__block_size)

    @staticmethod
    def __divide_into_blocks(image: np.ndarray, block_size: int = 8):
        height, width = image.shape
        blocks = [
            image[i : i + block_size, j : j + block_size]
            for i in range(0, height, block_size)
            for j in range(0, width, block_size)
        ]
        return blocks

    def __dct(self, blocks):
        return [self.__dct_2d(block) for block in blocks]

    def __inverse_dct(self, blocks):
        return [self.__inverse_dct_2d(block) for block in blocks]

    def __dct_2d(self, block: np.ndarray) -> np.ndarray:
        return self.__transform_matrix @ block @ self.__transform_matrix_transposed

    def __inverse_dct_2d(self, block: np.ndarray) -> np.ndarray:
        return self.__transform_matrix_transposed @ block @ self.__transform_matrix

    @staticmethod
    def quantize_blocks(blocks, quantization_factor):
        return [
            np.round(block / quantization_factor) * quantization_factor
            for block in blocks
        ]

    @staticmethod
    def reconstruct_image(blocks, image_shape, block_size=8):
        h, w = image_shape
        reconstructed = np.zeros((h, w))
        index = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                reconstructed[i : i + block_size, j : j + block_size] = blocks[index]
                index += 1
        return reconstructed
