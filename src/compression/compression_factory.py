from compression import Compression
from compression.algorithms.dct import DCTCompression
from compression.algorithms.wavelet import HaarWaveletCompression


class CompressionFactory:
    """
    Factory class for creating compression algorithm instances.
    """

    @staticmethod
    def create_compression(algorithm: str, **kwargs) -> Compression:
        """
        Create a compression algorithm instance.

        Args:
            algorithm (str): The name of the compression algorithm ('dct' or 'wavelet').
            **kwargs: Parameters specific to the compression algorithm.

        Returns:
            Compression: An instance of the specified compression algorithm.

        Raises:
            ValueError: If the specified algorithm is not recognized.
        """
        algorithm = algorithm.lower()
        if algorithm == "dct":
            return DCTCompression(**kwargs)
        elif algorithm == "haarwavelet":
            return HaarWaveletCompression(**kwargs)
        else:
            raise ValueError(f"Unknown compression algorithm: {algorithm}")
