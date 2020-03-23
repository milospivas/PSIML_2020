"""
    Task D - WORLD MAP

    Given a "large" image and a set of smaller patches (of constant dimensions)
    find x,y coordinates of each smaller patch in the large image.
    Patches can be exact copies or filtered in some way.
"""

from PIL import Image
import numpy as np
from scipy import signal, fftpack
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def input_read(in_path : str = None) -> (str, int, int, int, list):
    'Read the input'
    with open(in_path) as f:                           # TODO comment
        lines = [line.rstrip() for line in f]       # TODO comment
    
    wmap_path = set_root + lines[0].split("@@")[-1] 
    # wmap_path = input()        # TODO uncomment

    n_patches = int(lines[1])
    w, h = lines[2].split(" ")
    # n_patches = int(input())      # TODO uncomment
    # w, h = input().split(" ")     # TODO uncomment
    w, h = int(w), int(h)

    for i in range(3, len(lines)):
        lines[i] = set_root + lines[i].split("@@")[-1] 

    return wmap_path, n_patches, w, h, lines[3:]


def sol_read(out_path : str) -> np.array:
    with open(out_path) as out:
        out_lines = [line.rstrip() for line in out]
        sol_xy = np.empty((len(out_lines), 2))
        for j, line in enumerate(out_lines):
            x,y = np.array(line.split(" "))
            x = int(x)
            y = int(y)
            sol_xy[j, :] = np.array((x,y))

    return sol_xy


def wmap_preprocess(wmap : np.array, H : int, W : int, h : int, w : int, bin_cnt : int = 128) -> np.array:
    'Preprocess the world map. Transform it into 3D array of (H - h + 1) by (W - w + 1) flattened and normalized arrays.'
    # 'Preprocess the world map. Calculate and return histograms for every subpatch.'
    # h = h//2
    # w = w//2
    m = H - h + 1
    n = W - w + 1

    wmap_flats = np.empty((m, n, h*w))
    sigmas = np.empty((m,n))

    for y in range(m):
        for x in range(n):
            wmap_flats[y,x,:] = wmap[y : y+h, x : x+w].ravel()
            mu = np.mean(wmap_flats[y,x,:])
            std = np.std(wmap_flats[y,x,:])
            if std != 0:
                wmap_flats[y,x,:] -= mu
                wmap_flats[y,x,:] /= std
            else:
                wmap_flats[y,x,:] = 0
            sigmas[y, x] = std

            # if x == 103//2 and y == 71//2:
            #     plt.figure()
            #     plt.imshow(wmap, cmap='gray')

            #     plt.figure()
            #     plt.imshow(wmap[y : y+h, x : x+w], cmap='gray')

            #     plt.figure()
            #     plt.imshow(wmap_flats[y, x , :].reshape(h, w), cmap='gray')

            #     plt.show()
            #     print("", end="")

    return wmap_flats, sigmas

inputs = [
            "D/public/inputs/0.txt"
            ,"D/public/inputs/1.txt"
            ,"D/public/inputs/2.txt"
            ,"D/public/inputs/3.txt"
            ,"D/public/inputs/4.txt"
            ,"D/public/inputs/5.txt"
            ,"D/public/inputs/6.txt"
            ,"D/public/inputs/7.txt"
            ,"D/public/inputs/8.txt"
            ,"D/public/inputs/9.txt"
        ]

outputs = [
             "D/public/outputs/0.txt"
            ,"D/public/outputs/1.txt"
            ,"D/public/outputs/2.txt"
            ,"D/public/outputs/3.txt"
            ,"D/public/outputs/4.txt"
            ,"D/public/outputs/5.txt"
            ,"D/public/outputs/6.txt"
            ,"D/public/outputs/7.txt"
            ,"D/public/outputs/8.txt"
            ,"D/public/outputs/9.txt"
        ]


# main -------------------------------------------------------------------

# if __name__ == "__main__":    # TODO uncomment, indent bellow
    # for i in n_patches():

start = 4
stop = 10
set_root = "D/public/set"
for i_in, in_path in enumerate(inputs[start:stop]):
    i = i_in + start
    print("Set: ", i)
    sol_xy = sol_read(outputs[i])

    wmap_path, n_patches, w, h, patches_paths = input_read(in_path)
    # wmap_path, n_patches, w, h, patches_paths = input_read()   # TODO uncomment

    wmap_file = Image.open(wmap_path)
    W, H = wmap_file._size

    wmap_file_g = wmap_file.convert('L') # converting to grayscale

    # working only with grayscale images
    wmap = np.array(wmap_file_g)

    # size of the map
    H = len(wmap)
    W = len(wmap[0])

    # scaling parameters
    d_factor = 2
    H_d = H//d_factor
    W_d = W//d_factor
    h_d = h//d_factor
    w_d = w//d_factor

    # Preprocessing
    # resizing
    wmap_file_g_d = wmap_file_g.resize((W_d, H_d))
    wmap_d = np.array(wmap_file_g_d)

    # flattening and normalizing
    wmap_flats, wmap_sigmas = wmap_preprocess(wmap_d, H_d, W_d, h_d, w_d)
    
    m_d = H_d - h_d + 1
    n_d = W_d - w_d + 1

    R = np.zeros((m_d, n_d))

    start = 0
    for p_it, patch_path in enumerate(patches_paths[start:]):
        p = p_it + start
        print("Patch number:", p)
        patch_file = Image.open(patch_path)
        
        patch_file_g = patch_file.convert('L')  # converting to grayscale

        patch_file_g_d = patch_file_g.resize((h_d, w_d))

        # # working only with grayscale image
        patch = np.array(patch_file_g)
        # patch_filt = gaussian_filter(patch, sigma = 0.1)

        patch_d = np.array(patch_file_g_d)

        # plotting
        # plt.figure()
        # plt.imshow(patch, cmap='gray')
        # plt.title("OG search patch")

        # plt.figure()
        # plt.imshow(patch_d, cmap='gray')
        # plt.title("Downscaled search patch")


        # patch normalization
        mu = np.mean(patch_d)
        std = np.std(patch_d)
        if std != 0:
            patch_d = (patch_d - mu) / std
        else:
            patch_d[:,:] = 0

        # plotting after normalization
        # plt.figure()
        # plt.imshow(patch_d, cmap='gray')
        # plt.title("Downscaled and normalized search patch")

        # # test crosscorrelation:
        # x,y = sol_xy[p, :]
        # x = int(x)//d_factor
        # y = int(y)//d_factor
        # wmp = wmap_flats[y, x, :].reshape(h_d, w_d)

        # plt.figure()
        # plt.imshow(wmp, cmap='gray')
        # plt.title("Correct Solution world map patch")

        # plt.figure()
        # plt.imshow(wmp*patch_d, cmap='gray')
        # plt.title("Normalized cross-correlation")

        # plt.show()

        # computing the sum of square differences
        # R = np.sum(np.square(wmap_flats - patch_d.ravel()), axis=2)
        
        # calculating cross-correlation
        R = np.dot(wmap_flats, patch_d.ravel())/(h_d * w_d)

        # idx = R.argmin()
        idx = R.argmax()
        R_w = len(R[0])
        x_d = idx % R_w
        y_d = idx // R_w
        x = x_d * d_factor
        y = y_d * d_factor

        print(x, y)

        # plt.figure()
        # plt.imshow(wmap, cmap='gray')
        # plt.plot(x, y, 'bo', ms='6')
        # plt.title("WMAP")

        # plt.figure()
        # plt.imshow(patch, cmap='gray')
        # plt.title("Patch")

        # plt.figure()
        # plt.imshow(wmap_d, cmap='gray')
        # plt.title("WMAP downscaled")

        # plt.figure()
        # plt.imshow(patch_d, cmap='gray')
        # plt.title("Patch downscaled")

        # plt.figure()
        # plt.imshow(wmap_patch_conv, cmap='gray')
        # plt.title("Convolution")

        # plt.figure()
        # plt.imshow(wmap_patch_xcorr, cmap='gray')
        # plt.title("Xcorr")

        # plt.figure()
        # plt.imshow(R, cmap='gray')
        # plt.title("Corrcoef")

        # plt.show()
        print()

                