import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

# X = misc.face(gray=True)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()


def ex1_1(n1, n2):
    return np.sin(2*np.pi*n1 + 3*np.pi*n2)


def ex1_2(n1, n2):
    return np.sin(4*np.pi*n1) + np.cos(6*np.pi*n2)


def ex1_3(m1, m2):
    N = len(m1)
    arr = np.zeros(shape=(N, N))
    for elem1 in m1:
        for elem2 in m2:
            if elem1 == 0 and (elem2 == 5 or elem2 == N-5):
                arr[elem1][elem2] = 1
            else:
                arr[elem1][elem2] = 0
    return arr

def ex1_4(m1, m2):
    N = len(m1)
    arr = np.zeros(shape=(N,N))
    for elem1 in m1:
        for elem2 in m2:
            if elem2 == 0 and (elem1 == 5 or elem1 == N-5):
                arr[elem1][elem2] = 1
            else:
                arr[elem1][elem2] = 0
    return arr

def ex1_5(m1, m2):
    N = len(m1)
    arr = np.zeros(shape=(N, N))
    for elem1 in m1:
        for elem2 in m2:
            if elem1 == elem2 == 5:
                arr[elem1][elem2] = 1
            elif elem1 == elem2 == N-5:
                arr[elem1][elem2] = 1
            else:
                arr[elem1][elem2] = 0
    return arr


def ex1():
    n1 = np.arange(0,101,1,dtype=int)
    n2 = np.arange(0,101,1,dtype=int)

    image_ex1_3 = ex1_3(n1, n2)
    image_ex1_4 = ex1_5(n1, n2)
    image_ex1_5 = ex1_5(n1, n2)
    n1, n2 = np.meshgrid(n1, n2)
    image_ex1_1 = ex1_1(n1, n2)
    image_ex1_2 = ex1_2(n1, n2)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_ex1_1, cmap='viridis')
    plt.title('Imaginea pentru ex1_1')
    plt.subplot(1, 2, 2)
    plt.imshow(image_ex1_2, cmap='viridis')
    plt.title('Imaginea pentru ex1_2')
    plt.show()

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_ex1_3, cmap='viridis')
    plt.title('Imaginea pentru ex1_3')
    plt.subplot(1, 3, 2)
    plt.imshow(image_ex1_4, cmap='viridis')
    plt.title('Imaginea pentru ex1_4')
    plt.subplot(1, 3, 3)
    plt.imshow(image_ex1_5, cmap='viridis')
    plt.title('Imaginea pentru ex1_5')
    plt.show()

    # Spectrum
    spectrum_ex1_1 = np.abs(np.fft.fft2(image_ex1_1))
    spectrum_ex1_2 = np.abs(np.fft.fft2(image_ex1_2))
    spectrum_ex1_3 = np.abs(np.fft.fft2(image_ex1_3))
    spectrum_ex1_4 = np.abs(np.fft.fft2(image_ex1_4))
    spectrum_ex1_5 = np.abs(np.fft.fft2(image_ex1_5))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_1 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_1')
    plt.subplot(1, 2, 2)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_2 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_2')
    plt.show()

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_3 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_3')
    plt.subplot(1, 3, 2)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_4 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_4')
    plt.subplot(1, 3, 3)
    plt.imshow(np.fft.fftshift(np.log(spectrum_ex1_5 + 1)), cmap='viridis')
    plt.title('Spectrul pentru ex1_5')
    plt.show()


def ex2_3():
    # Exercitiul 2.
    image = misc.face(gray=True)
    Y = np.fft.fft2(image)
    freq_db = 20 * np.log10(np.abs(Y))
    freq_cutoff = 140

    Y_cutoff = Y.copy()
    Y_cutoff[freq_db > freq_cutoff] = 0

    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)
    compressed_img = X_cutoff

    # Afișează imaginile
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Imaginea originală")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Imaginea comprimată")
    plt.imshow(compressed_img, cmap='gray')

    plt.show()
    
    # Exercitiul 2.
    img = misc.face(gray=True)

    f_transform = np.fft.fft2(img)
    power_spectrum = np.abs(f_transform) ** 2
    zgomot = np.mean(power_spectrum)
    snr_threshold = 10
    cutoff = snr_threshold * zgomot
    f_transform_filtered = f_transform * (power_spectrum > cutoff)
    compressed_img = np.real(np.fft.ifft2(f_transform_filtered))

    # Exercitiul 3.
    snr_original = 10 * np.log10(np.mean(img ** 2) / np.mean((img - compressed_img) ** 2))
    restored_img = img - img + compressed_img
    snr_restored = 10 * np.log10(np.mean(img ** 2) / np.mean((img - restored_img) ** 2))

    # Afișează imaginile
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Imaginea originală")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Imaginea comprimată")
    plt.imshow(compressed_img, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Imaginea restaurată")
    plt.imshow(restored_img, cmap='gray')

    plt.show()

    # Afișează raportul SNR înainte și după
    print(f"Raportul SNR înainte de comprimare: {snr_original:.2f} dB")
    print(f"Raportul SNR după eliminarea zgomotului: {snr_restored:.2f} dB")


if __name__ == "__main__":
    ex1()
    # ex2_3()
