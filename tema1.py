import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.metrics import mean_squared_error
import cv2 as cv
import os


## Cod din laborator
"""
X = misc.ascent()

Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down);

plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('Down-sampled')
plt.show()

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# Encoding
x = X[:8, :8]
y = dctn(x)
y_jpeg = Q_jpeg*np.round(y/Q_jpeg)

# Decoding
x_jpeg = idctn(y_jpeg)

# Results
y_nnz = np.count_nonzero(y)
y_jpeg_nnz = np.count_nonzero(y_jpeg)

plt.subplot(121).imshow(x, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
plt.title('JPEG')
plt.show()

print('Componente în frecvență:' + str(y_nnz) +
      '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))
"""


## Exercitiul 1.

X = misc.ascent()
Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 28, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])
X_jpeg = X.copy()
modified_image = np.zeros_like(X_jpeg, dtype=float)

for i in range(0, X_jpeg.shape[0], 8):
    for j in range(0, X_jpeg.shape[1], 8):
        patch = X_jpeg[i:i+8, j:j+8]
        # Encoding
        y = dctn(patch)
        y_jpeg = Q_jpeg * np.round(y / Q_jpeg)
        # Decoding
        patch_jpeg = idctn(y_jpeg)
        # Imaginea finala
        modified_image[i:i+8, j:j+8] = patch_jpeg

plt.subplot(121).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(modified_image, cmap=plt.cm.gray)
plt.title('JPEG')
plt.show()


## Exercitiul 2.

X_color = misc.face()
image_ycbcr = rgb2ycbcr(X_color)

# Cele 3 canale ale YCbCr
modified_y = np.zeros_like(image_ycbcr[:, :, 0], dtype=float)
modified_cb = np.zeros_like(image_ycbcr[:, :, 1], dtype=float)
modified_cr = np.zeros_like(image_ycbcr[:, :, 2], dtype=float)

for channel_index, channel in enumerate([modified_y, modified_cb, modified_cr]):
    for i in range(0, image_ycbcr.shape[0], 8):
        for j in range(0, image_ycbcr.shape[1], 8):
            patch = image_ycbcr[i:i+8, j:j+8, channel_index]
            # Encoding
            y = dctn(patch)
            y_jpeg = Q_jpeg * np.round(y / Q_jpeg)
            # Decoding
            patch_jpeg = idctn(y_jpeg)
            # Imaginea finala
            channel[i:i+8, j:j+8] = patch_jpeg

# Y'CbCr -> RGB
modified_image_ycbcr = np.stack([modified_y, modified_cb, modified_cr], axis=-1)
modified_X_color = ycbcr2rgb(modified_image_ycbcr)

plt.subplot(121).imshow(X_color)
plt.title('Original')
plt.subplot(122).imshow(modified_X_color)
plt.title('JPEG')
plt.show()


## Exercitiul 3.

X = misc.ascent()
Y2 = dctn(X, type=2)
k = 150
Y_ziped = Y2.copy()
Y_ziped[k:] = 0
X_ziped = idctn(Y_ziped)

prag_MSE = 5
mse = np.mean((X - X_ziped) ** 2)

while mse > prag_MSE:
    print(f"MSE este prea mare: {mse}. Ajustare compresie...")
    k += 5

    Y_ziped = Y2.copy()
    Y_ziped[k:] = 0
    X_ziped = idctn(Y_ziped)

    mse = np.mean((X - X_ziped) ** 2)


plt.imshow(X_ziped, cmap=plt.cm.gray)
plt.show()

print(f"MSE final: {mse}")


## Exercitiul 4.

def compress_image(image):
    X = image.copy()
    Y2 = dctn(X, type=2)

    k = 50
    Y_zipped = Y2.copy()
    Y_zipped[k:] = 0
    X_zipped = idctn(Y_zipped)

    prag_MSE = 2.5
    mse = np.mean((X - X_zipped) ** 2)

    while mse > prag_MSE:
        #print(f"MSE is too high: {mse}. Adjusting compression...")
        k += 5
        Y_zipped = Y2.copy()
        Y_zipped[k:] = 0
        X_zipped = idctn(Y_zipped)

        mse = np.mean((X - X_zipped) ** 2)
    #print("MSE = ", mse)
    return X_zipped.astype(np.uint8)


def extract_frames_and_compress(video_path, compressed_video_path):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(compressed_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame {}".format(frame_number))
            break

        compressed_frame = compress_image(frame)

        out.write(compressed_frame)

    cap.release()
    out.release()

    print("Compressed video created successfully.")

# Example usage
video_path = 'test_video.mp4'
compressed_video_path = "compressed_video.mp4"

extract_frames_and_compress(video_path, compressed_video_path)


