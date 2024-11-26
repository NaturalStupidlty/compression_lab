import numpy as np

from compression import Compression


class HaarWaveletCompression(Compression):
    def compress(
        self, image: np.ndarray, level: int = 1, quantization_factor: int = 10, **kwargs
    ):
        coefficients = self.wavelet_decompose(image, level)
        return self.quantize_coeffs(coefficients, quantization_factor)

    def decompress(self, coefficients, **kwargs):
        return self.wavelet_reconstruct(coefficients)

    @staticmethod
    def wavelet_decompose(image, level):
        coefficients = []
        current_image = image.astype(np.float32)
        for _ in range(level):
            LL = (
                current_image[::2, ::2]
                + current_image[1::2, ::2]
                + current_image[::2, 1::2]
                + current_image[1::2, 1::2]
            ) / 4
            LH = (
                current_image[::2, ::2]
                - current_image[1::2, ::2]
                + current_image[::2, 1::2]
                - current_image[1::2, 1::2]
            ) / 4
            HL = (
                current_image[::2, ::2]
                + current_image[1::2, ::2]
                - current_image[::2, 1::2]
                - current_image[1::2, 1::2]
            ) / 4
            HH = (
                current_image[::2, ::2]
                - current_image[1::2, ::2]
                - current_image[::2, 1::2]
                + current_image[1::2, 1::2]
            ) / 4

            coefficients.append((LH, HL, HH))
            current_image = LL

        coefficients.append(current_image)
        return coefficients

    @staticmethod
    def quantize_coeffs(coeffs, quantization_factor):
        quantized_coeffs = [coeffs[-1]]
        for detail_level in reversed(coeffs[:-1]):
            quantized_level = tuple(
                np.round(detail / quantization_factor) * quantization_factor
                for detail in detail_level
            )
            quantized_coeffs.insert(0, quantized_level)
        return quantized_coeffs

    @staticmethod
    def wavelet_reconstruct(coeffs):
        current_image = coeffs[-1]
        for detail_level in reversed(coeffs[:-1]):
            LH, HL, HH = detail_level
            height, width = LH.shape
            LL = current_image

            current_image = np.zeros((height * 2, width * 2), dtype=np.float32)
            current_image[::2, ::2] = LL + LH + HL + HH
            current_image[1::2, ::2] = LL - LH + HL - HH
            current_image[::2, 1::2] = LL + LH - HL - HH
            current_image[1::2, 1::2] = LL - LH - HL + HH

        return current_image
