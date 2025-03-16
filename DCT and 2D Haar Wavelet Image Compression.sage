### GRAYSCALE VECTOR COMPRESSION

# Define the basis vectors
v1 = vector([1, 1, 1, 1, 1, 1, 1, 1])
v2 = vector([1, 1, 1, 1, -1, -1, -1, -1])
v3 = vector([1, 1, -1, -1, 0, 0, 0, 0])
v5 = vector([1, -1, 0, 0, 0, 0, 0, 0])
v7 = vector([0, 0, 0, 0, 1, -1, 0, 0])

# Compute the linear combinations
scaled_v1 = v1  # v1 only
combo1 = 100 * v1 + 50 * v2 # second vector
combo2 = 128 * v1 - 64 * v3 + 32 * v5 - 16 * v7 # third vector

# Convert vectors to 1x8 matrices
mat_v1 = matrix(1, 8, scaled_v1.list())
mat_combo1 = matrix(1, 8, combo1.list())
mat_combo2 = matrix(1, 8, combo2.list())

# Plot the matrices as grayscale images
plot_v1 = matrix_plot(mat_v1, cmap='gray', vmin=0, vmax=255)
plot_combo1 = matrix_plot(mat_combo1, cmap='gray', vmin=0, vmax=255)
plot_combo2 = matrix_plot(mat_combo2, cmap='gray', vmin=0, vmax=255)

# Display the plots
show(plot_v1)      # Uniform almost black row
show(plot_combo1)  # Left half brighter, right half darker
show(plot_combo2)  # Varying intensities

# Define the vectors
v = [19, 147, 9, 161, 230, 54, 201, 6]
v1 = [20, 148, 8, 160, 230, 54, 201, 6]
v2 = [39.375, 167.375, 27.375, 179.375, 191.375, 15.375, 200.875, 5.875]
v3 = [103.375, 103.375, 103.375, 103.375, 191.375, 15.375, 200.875, 5.875]

# Convert to 1x8 matrices
mat_v = matrix(1, 8, v)
mat_v1 = matrix(1, 8, v1)
mat_v2 = matrix(1, 8, v2)
mat_v3 = matrix(1, 8, v3)

# Create grayscale plots
plot_v = matrix_plot(mat_v, cmap='gray', vmin=0, vmax=255)
plot_v1 = matrix_plot(mat_v1, cmap='gray', vmin=0, vmax=255)
plot_v2 = matrix_plot(mat_v2, cmap='gray', vmin=0, vmax=255)
plot_v3 = matrix_plot(mat_v3, cmap='gray', vmin=0, vmax=255)

# Display the plots
print("v:")
show(plot_v)   # Original image
print("Compressed with epsilon=5")
show(plot_v1)  # Compressed with epsilon=5
print("Compressed with epsilon=25")
show(plot_v2)  # Compressed with epsilon=25
print("Compressed with epsilon=80")
show(plot_v3)  # Compressed with epsilon=80

# Define the basis vectors
v1 = vector([1, 1, 1, 1, 1, 1, 1, 1])
v2 = vector([1, 1, 1, 1, -1, -1, -1, -1])
v3 = vector([1, 1, -1, -1, 0, 0, 0, 0])
v4 = vector([0, 0, 0, 0, 1, 1, -1, -1])
v5 = vector([1, -1, 0, 0, 0, 0, 0, 0])
v6 = vector([0, 0, 1, -1, 0, 0, 0, 0])
v7 = vector([0, 0, 0, 0, 1, -1, 0, 0])
v8 = vector([0, 0, 0, 0, 0, 0, 1, -1])

# Define the original coordinates of v in the basis S
coords_v = [103.375, -19.375, -1, 19.25, -64, -76, 88, 97.5]

# Convert v to the standard basis for reference
v_standard = coords_v[0]*v1 + coords_v[1]*v2 + coords_v[2]*v3 + coords_v[3]*v4 + coords_v[4]*v5 + coords_v[5]*v6 + coords_v[6]*v7 + coords_v[7]*v8
mat_v = matrix(1, 8, v_standard.list())

# List of epsilon values to experiment with
epsilon_values = [1, 5, 10, 20, 30, 50, 80, 100]

# Plot the original vector
print("Original v:")
matrix_plot(mat_v, fontsize=10, cmap="gray", vmin=0, vmax=256).show()

# Iterate over each epsilon value
for eps in epsilon_values:
    # Compress the coordinates by setting values with abs < eps to 0
    compressed_coords = [0 if abs(val) < eps else val for val in coords_v]
    
    # Convert the compressed coordinates back to the standard basis
    v_compressed = (compressed_coords[0]*v1 + compressed_coords[1]*v2 + 
                    compressed_coords[2]*v3 + compressed_coords[3]*v4 + 
                    compressed_coords[4]*v5 + compressed_coords[5]*v6 + 
                    compressed_coords[6]*v7 + compressed_coords[7]*v8)
    
    # Convert to a 1x8 matrix for plotting
    mat_compressed = matrix(1, 8, v_compressed.list())
    
    # Print the compressed coordinates and plot
    print(f"\n-----------------\nEpsilon = {eps}")
    print("Compressed coordinates:", compressed_coords)
    matrix_plot(mat_compressed, fontsize=10, cmap="gray", vmin=0, vmax=256).show()


### DCT IMAGE COMPRESSION
from PIL import Image
#from urllib.request import urlopen
import numpy as np
from scipy.fftpack import dct, idct

# Define 2D DCT and inverse DCT functions
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Load the capybara image and convert to grayscale
img = Image.open("Capy.jpg").convert('L')  # 'L' mode for grayscale
img_array = np.array(img)

# Display the original image
print("Original Capybara Image")
display(img)

# Apply 2D DCT to the entire image
dct_img = dct2(img_array)

# Define a range of epsilon values to test compression
epsilons = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000]

# Function to compress by setting DCT coefficients < epsilon to 0
def compress_dct(dct_img, epsilon):
    compressed_dct = np.where(np.abs(dct_img) < epsilon, 0, dct_img)
    return compressed_dct

# Process and display compressed images for each epsilon
for eps in epsilons:
    # Compress the DCT coefficients
    compressed_dct = compress_dct(dct_img, eps)
    
    # Reconstruct the image using inverse DCT
    compressed_img = idct2(compressed_dct)
    
    # Clip values to valid pixel range [0, 255] and convert to uint8
    compressed_img = np.clip(compressed_img, 0, 255).astype('uint8')
    
    # Display the compressed image
    print(f"Compressed Capybara Image with Epsilon = {eps}")
    display(Image.fromarray(compressed_img))

# Calculate and display the fraction of non-zero coefficients
for eps in epsilons:
    compressed_dct = compress_dct(dct_img, eps)
    num_nonzero = np.count_nonzero(compressed_dct)
    total_pixels = compressed_dct.size
    fraction_nonzero = num_nonzero / total_pixels
    print(f"Epsilon = {eps}, Fraction of non-zero coefficients: {fraction_nonzero:.5f}")

# HAAR WAVELET IMAGE COMPRESSION


from PIL import Image
import numpy as np
import pywt  # Requires PyWavelets

# Load the capybara image and convert to grayscale
img = Image.open("Capy.jpg").convert('L')  # 'L' for grayscale
img_array = np.array(img)

# Display the original image
print("Original Capybara Portrait")
display(img)

# Define epsilon values for compression
epsilons = [0, 25, 50, 65, 70, 75, 80, 85, 90, 95, 100, 125]

# Apply 2D Haar wavelet transform (multi-level decomposition)
coeffs = pywt.wavedec2(img_array, 'haar', level=5)  # Decompose to level 5

# Function to compress wavelet coefficients by thresholding
def compress_wavelet(coeffs, epsilon):
    compressed_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):  # Detail coefficients (horizontal, vertical, diagonal)
            compressed_c = tuple(np.where(np.abs(detail) < epsilon, 0, detail) for detail in c)
        else:  # Approximation coefficients
            compressed_c = np.where(np.abs(c) < epsilon, 0, c)
        compressed_coeffs.append(compressed_c)
    return compressed_coeffs

# Lists to store compressed images and fraction of non-zero coefficients
compressed_images = []
fraction_nonzero = []

# Process each epsilon value
for eps in epsilons:
    # Compress the wavelet coefficients
    compressed_coeffs = compress_wavelet(coeffs, eps)
    
    # Reconstruct the image
    compressed_img = pywt.waverec2(compressed_coeffs, 'haar')
    compressed_img = np.clip(compressed_img, 0, 255).astype('uint8')  # Ensure valid pixel values
    
    # Store the compressed image
    compressed_images.append(compressed_img)
    
    # Calculate fraction of non-zero coefficients
    num_nonzero = sum(np.count_nonzero(c) if not isinstance(c, tuple) else sum(np.count_nonzero(detail) for detail in c) for c in compressed_coeffs)
    total_coeffs = sum(c.size if not isinstance(c, tuple) else sum(detail.size for detail in c) for c in compressed_coeffs)
    fraction = num_nonzero / total_coeffs
    fraction_nonzero.append(round(fraction, 5))

# Convert compressed images to PIL Image objects
img_objects = [Image.fromarray(img) for img in compressed_images]

# Display images in rows (4 rows of 3, adjusting for 12 epsilons)
rows = [
    Image.fromarray(np.hstack(compressed_images[:3])),
    Image.fromarray(np.hstack(compressed_images[3:6])),
    Image.fromarray(np.hstack(compressed_images[6:9])),
    Image.fromarray(np.hstack(compressed_images[9:12]))
]

# Display results
for r in range(len(rows)):
    start_idx = r * 3
    end_idx = min(start_idx + 3, len(epsilons))  # Handle the last row with 1 image
    print("Epsilon values:", epsilons[start_idx:end_idx],
          "Fraction of non-zero coefficients:", fraction_nonzero[start_idx:end_idx])
    display(rows[r])

# Handle the last epsilon value separately if not included in rows
if len(epsilons) > 12:
    print("Epsilon value:", epsilons[12],
          "Fraction of non-zero coefficients:", fraction_nonzero[12])
    display(img_objects[12])
