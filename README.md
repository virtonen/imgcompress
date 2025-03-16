# imgcompress
Lossy Compression Project: From 8 grayscale pixels to an image

---

## DCT and 2D Haar Wavelet Image Compression

This SageMath script uses the Discrete Cosine Transform (DCT) and 2D Haar Wavelet Transform for lossy image compression. The script is divided into three main sections: Grayscale Vector Compression, DCT Image Compression, and Haar Wavelet Image Compression.

### Grayscale Vector Compression

- **Basis Vectors**: Defines a set of basis vectors for grayscale compression.
- **Linear Combinations**: Computes linear combinations of these vectors.
- **Matrix Conversion**: Converts vectors to matrices and plots them as grayscale images.
- **Compression with Epsilon Values**: Demonstrates the effect of different epsilon values on compression by setting vector components with absolute values less than epsilon to zero and plotting the results.

### DCT Image Compression

- **Image Loading**: Loads and converts an image to grayscale.
- **2D DCT Application**: Applies 2D DCT to the image.
- **Compression by Thresholding**: Compresses the image by setting DCT coefficients below certain epsilon values to zero.
- **Reconstruction and Display**: Reconstructs and displays the compressed images for different epsilon values, showing the fraction of non-zero coefficients.

### Haar Wavelet Image Compression

- **Image Loading**: Loads and converts an image to grayscale.
- **2D Haar Wavelet Transform**: Applies 2D Haar wavelet transform for multi-level decomposition.
- **Compression by Thresholding**: Compresses the wavelet coefficients by setting values below epsilon to zero.
- **Reconstruction and Display**: Reconstructs and displays the compressed images for different epsilon values, showing the fraction of non-zero coefficients.

### Usage

To run the compression algorithms, load the script in SageMath and execute it with your chosen image. The script will display the original and compressed images along with the details of the compression process.

---
