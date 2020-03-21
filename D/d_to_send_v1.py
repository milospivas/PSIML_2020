"""
    Task D - WORLD MAP
    
    Given a "large" image and a set of smaller patches (of constant dimensions)
    find x,y coordinates of each smaller patch in the large image.
    Patches can be exact copies or filtered in some way.
"""

from PIL import Image
import numpy as np
# from scipy import signal, fftpack
# from scipy.ndimage.filters import gaussian_filter
# import matplotlib.pyplot as plt

def input_read(in_path : str = None) -> (str, int, int, int):
    'Read the input'
    # with open(in_path) as f:
    #     lines = [line.rstrip() for line in f]
    
    # wmap_path = set_root + lines[0].split("@@")[-1] 
    wmap_path = input()

    # n_patches = int(lines[1])
    # w, h = lines[2].split(" ")
    n_patches = int(input())
    w, h = input().split(" ")
    w, h = int(w), int(h)

    # for i in range(3, len(lines)):
    #     lines[i] = set_root + lines[i].split("@@")[-1] 

    return wmap_path, n_patches, w, h #, lines[3:]

# def sol_read(out_path : str) -> np.array:
#     with open(out_path) as out:
#         out_lines = [line.rstrip() for line in out]
#         sol_xy = np.empty((len(out_lines), 2))
#         for j, line in enumerate(out_lines):
#             x,y = np.array(line.split(" "))
#             x = int(x)
#             y = int(y)
#             sol_xy[j, :] = np.array((x,y))

#     return sol_xy


def wmap_preprocess(wmap : np.array, H : int, W : int, h : int, w : int, bin_cnt : int = 128) -> np.array:
    'Preprocess the world map. Transform it into 3D array of (H - h + 1) by (W - w + 1) flattened and normalized arrays.'
    # 'Preprocess the world map. Calculate and return histograms for every subpatch.'
    m = H - h + 1
    n = W - w + 1

    wmap_flats = np.empty((m, n, h*w))
    # sigmas = np.empty((m,n))

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
            # sigmas[y, x] = std

    return wmap_flats #, sigmas

# def patch_preprocess(patch : np.array, h : int, w : int, bin_cnt : int = 128) -> (np.array, np.array, np.array, np.array):
#     'Subdivide the patch into 4 quadrants and calcualte histograms.'
#     h_sub = h//2
#     w_sub = w//2
#     q00_hist, _ = np.histogram(patch[:h_sub, :w_sub], bins = bin_cnt)
#     q01_hist, _ = np.histogram(patch[:h_sub, w_sub:], bins = bin_cnt)
#     q10_hist, _ = np.histogram(patch[h_sub:, :w_sub], bins = bin_cnt)
#     q11_hist, _ = np.histogram(patch[h_sub:, w_sub:], bins = bin_cnt)

#     return q00_hist, q01_hist, q10_hist, q11_hist

# def corrcoef(x_array, y_array, x_std, y_std):
#     if (x_std == 0) and (y_std == 0):
#         R = 1
#     elif ((x_std == 0) and (y_std != 0)) or ((x_std != 0) and (y_std == 0)):
#         R = -1
#     else:
#         R = np.corrcoef(x_array, y_array)[0,1]
    
#     return R

# def corrcoefs2d(X, Y):
#     H = len(X)
#     W = len(X[0])

#     h = len(Y)
#     w = len(Y[0])

#     m = H - h + 1
#     n = W - w + 1

#     xcorr = np.empty((m, n))

#     y_array = Y.ravel()
#     y_std = np.std(y_array)
#     # y00 = Y[ :h//2  , :w//2 ].ravel()
#     # y01 = Y[ :h//2  ,  w//2:].ravel()
#     # y10 = Y[  h//2: , :w//2 ].ravel()
#     # y11 = Y[  h//2: ,  w//2:].ravel()
#     # y00_std = np.std(y00)
#     # y01_std = np.std(y01)    
#     # y10_std = np.std(y10)
#     # y11_std = np.std(y11)

#     for y in range(m):
#         for x in range(n):
#             x_array = X[y:y+h, x:x+w].ravel()
#             x_std = np.std(x_array)

#             # x00 = X[ y        : y + h//2 , x        : x + w//2 ].ravel()
#             # x01 = X[ y        : y + h//2 , x + w//2 : x + w    ].ravel()
#             # x10 = X[ y + h//2 : y + h    , x        : x + w//2 ].ravel()
#             # x11 = X[ y + h//2 : y + h    , x + w//2 : x + w    ].ravel()

#             # x00_std = np.std(x00)
#             # x01_std = np.std(x01)
#             # x10_std = np.std(x10)
#             # x11_std = np.std(x11)
#             # A = corrcoef(x00, y00, x00_std, y00_std)
#             # B = corrcoef(x01, y01, x01_std, y01_std)
#             # C = corrcoef(x10, y10, x10_std, y10_std)
#             # D = corrcoef(x11, y11, x11_std, y11_std)
            
#             # # normalizing to [0, 1]
#             # A = (A + 1)/2
#             # B = (B + 1)/2
#             # C = (C + 1)/2
#             # D = (D + 1)/2

#             # # unifying
#             # R = 0
#             # if (A + B + C + D) != 0:
#             #     R = 4 * A * B * C * D / (A + B + C + D)
            
#             xcorr[y,x] = corrcoef(x_array, y_array, x_std, y_std)

#     return xcorr


# def draw_wp_hists(x_hist, y_hist, bin_centers, xy_xcorr = None):
#     x_max_f = x_hist.argmax()
#     y_max_f = y_hist.argmax()

#     plt.figure()
#     plt.plot(bin_centers, x_hist)
#     plt.title("WMAP patch histogram, max at bin: " + str(x_max_f))

#     plt.figure()
#     plt.plot(bin_centers, y_hist)
#     plt.title("Search patch histogram, max at bin: " + str(y_max_f))
    
#     if xy_xcorr is not None:
#         s = int(bin_centers[0])
#         n = len(xy_xcorr)
#         plt.figure()
#         plt.plot(range(s, s+n), xy_xcorr)
#         plt.title("Cross-correlation of two histograms")

#     plt.show()

# def sliding_corrcoef(x : np.array, y : np.array, s : int) -> float:
#     max_R = np.corrcoef(x,y)[0,1]
#     if abs(x.argmax() - y.argmax()) <= s:
#         for i in range(1,s//2):
#             R = np.corrcoef(y[i:], x[:-i])[0,1]
#             if R > max_R:
#                 max_R = R
#         for i in range(1,s//2):
#             R = np.corrcoef(x[i:], y[:-i])[0,1]
#             if R > max_R:
#                 max_R = R
    
#     return max_R

# def corrcoef_hists( y : int, x : int, h : int, w : int,
#                     wmap_hists : np.array,
#                     q00_hist : np.array,
#                     q01_hist : np.array,
#                     q10_hist : np.array,
#                     q11_hist : np.array
#                     ) -> float:
#     h = h//2
#     w = w//2
#     # calculating correlation coeficients
#     A = sliding_corrcoef(wmap_hists[y    , x    ], q00_hist, 16)
#     B = sliding_corrcoef(wmap_hists[y    , x + w], q01_hist, 16)
#     C = sliding_corrcoef(wmap_hists[y + h, x    ], q10_hist, 16)
#     D = sliding_corrcoef(wmap_hists[y + h, x + w], q11_hist, 16)
    
#     # normalizing to [0, 1]
#     A = (A + 1)/2
#     B = (B + 1)/2
#     C = (C + 1)/2
#     D = (D + 1)/2

#     # unifying
#     R = 0
#     if (A + B + C + D) != 0:
#         R = 4 * A * B * C * D / (A + B + C + D)
    
#     return R

# def compare(wmap, y, x, h, w, search_patch, draw_flag = False):
#     bin_cnt = 128
    
#     wmap_patch = wmap[y : y+h, x : x+w]

#     # wmap_patch = abs(fftpack.fft2(wmap_patch))
#     # patch = abs(fftpack.fft2(search_patch))
#     patch = search_patch


#     wmap_hist, wmap_bin_edges = np.histogram(wmap_patch, bins=bin_cnt)
#     wmap_bin_centers = 0.5 * (wmap_bin_edges[:-1] + wmap_bin_edges[1:])

#     patch_hist, patch_bin_edges = np.histogram(patch, bins=bin_cnt)
#     # patch_bin_centers = 0.5 * (patch_bin_edges[:-1] + patch_bin_edges[1:])

#     hist_xcorr = np.correlate(wmap_hist, patch_hist, 'full')

#     # Rxy = np.corrcoef(wmap_hist, patch_hist)
#     # R = Rxy[0,1]
#     R = sliding_corrcoef(wmap_hist, patch_hist, 16)
#     print("Histogram correlation coefficient:", R)

#     # xcorr = signal.correlate2d(wmap_patch, patch)
#     # conv = signal.fftconvolve(wmap_patch, patch)
#     # diff = np.square(wmap_patch - patch)

#     if R < 0.8 and draw_flag:
#     # if draw_flag:
#         fig = plt.figure()
#         plt.imshow(wmap_patch, cmap="gray")
#         plt.title("WMAP patch")

#         fig = plt.figure()
#         plt.imshow(patch, cmap="gray")
#         plt.title("search patch")

#         # fig = plt.figure()
#         # plt.imshow(xcorr, cmap="gray")
#         # plt.title("XCorr")

#         # fig = plt.figure()
#         # plt.imshow(conv, cmap="gray")
#         # plt.title("Convolution")

#         # fig = plt.figure()
#         # plt.imshow(diff, cmap="gray")
#         # plt.title("Square difference")

#         draw_wp_hists(wmap_hist, patch_hist, wmap_bin_centers, hist_xcorr)
#     pass


# inputs = [
#             "D/public/inputs/0.txt"
#             ,"D/public/inputs/1.txt"
#             ,"D/public/inputs/2.txt"
#             ,"D/public/inputs/3.txt"
#             ,"D/public/inputs/4.txt"
#             ,"D/public/inputs/5.txt"
#             ,"D/public/inputs/6.txt"
#             ,"D/public/inputs/7.txt"
#             ,"D/public/inputs/8.txt"
#             ,"D/public/inputs/9.txt"
#         ]

# outputs = [
#              "D/public/outputs/0.txt"
#             ,"D/public/outputs/1.txt"
#             ,"D/public/outputs/2.txt"
#             ,"D/public/outputs/3.txt"
#             ,"D/public/outputs/4.txt"
#             ,"D/public/outputs/5.txt"
#             ,"D/public/outputs/6.txt"
#             ,"D/public/outputs/7.txt"
#             ,"D/public/outputs/8.txt"
#             ,"D/public/outputs/9.txt"
#         ]


# main -------------------------------------------------------------------

if __name__ == "__main__":    # TODO uncomment, indent bellow

# start = 4
# stop = 10
# set_root = "D/public/set"
# for i_in, in_path in enumerate(inputs[start:stop]):
    # i = i_in + start
    # print("Set: ", i)
    # sol_xy = sol_read(outputs[i])

    # wmap_path, n_patches, w, h, patches_paths = input_read(in_path)
    wmap_path, n_patches, w, h = input_read()

    wmap_file = Image.open(wmap_path)
    W, H = wmap_file._size


    wmap_file_g = wmap_file.convert('L') # converting to grayscale

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
    wmap_flats = wmap_preprocess(wmap_d, H_d, W_d, h_d, w_d)
    
    # bounds for patch search
    m_d = H_d - h_d + 1
    n_d = W_d - w_d + 1

    # R = np.zeros((m_d, n_d))

    # start = 0
    # for p_it, patch_path in enumerate(patches_paths[start:]):
    for p in range(n_patches):
        patch_path = input()
        # p = p_it + start
        # print("Patch number:", p)
        patch_file = Image.open(patch_path)
        
        patch_file_g = patch_file.convert('L')  # converting to grayscale

        patch_file_g_d = patch_file_g.resize((h_d, w_d))

        # # working only with grayscale image
        patch = np.array(patch_file_g)
        # patch_filt = gaussian_filter(patch, sigma = 0.1)

        patch_d = np.array(patch_file_g_d)

        # # plotting
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

        # # plotting after normalization
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

        # computing sums of square differences
        # R = np.sum(np.square(wmap_flats - patch_d.ravel()), axis=2)
        
        # computing cross-correlations (not divided by length)
        R = np.dot(wmap_flats, patch_d.ravel())


        ## too slow
        # max_set = set()
        # for y in range(1,len(R)-1):
        #     for x in range(1, len(R[0])-1):
        #         if R[y,x] > 0.7 and R[y, x] > max(R[y - 1, x - 1],R[y - 1, x + 0],R[y - 1, x + 1],R[y + 0, x - 1],R[y + 0, x + 1],R[y + 1, x - 1],R[y + 1, x + 0],R[y + 1, x + 1]):
        #             max_set.add((x * d_factor,y * d_factor))

        # idx = R.argmin()
        idx = R.argmax()
        R_w = len(R[0])
        x = (idx % R_w) * d_factor
        y = (idx // R_w) * d_factor

        print(x, y)
        
        # # wmap_patch_conv = signal.fftconvolve(wmap_d, patch_d)         # everything is too slow!!!
        # # wmap_patch_xcorr = signal.correlate2d(wmap_d, patch_d)            # everything is too slow!!!
        # # wmap_patch_corrcoef = corrcoefs2d(wmap_d, patch_d)            # everything is too slow!!!
        # # wmap_patch_corrcoef = signal.correlate2d(wmap_d, patch_d)         # everything is too slow!!!
        # # idx = wmap_patch_corrcoef.argmax()
        # # W_xcorr = len(wmap_patch_corrcoef[0])
        # # x = idx % W_xcorr
        # # y = idx // W_xcorr
        # # x *= d_factor
        # # y *= d_factor

        # plt.figure()
        # plt.imshow(wmap, cmap='gray')
        # plt.plot(x, y, 'bo', ms='6')
        # plt.title("WMAP")

        # plt.figure()
        # plt.imshow(patch, cmap='gray')
        # plt.title("Patch")

        # # plt.figure()
        # # plt.imshow(wmap_d, cmap='gray')
        # # plt.title("WMAP downscaled")

        # # plt.figure()
        # # plt.imshow(patch_d, cmap='gray')
        # # plt.title("Patch downscaled")

        # # plt.figure()
        # # plt.imshow(wmap_patch_conv, cmap='gray')
        # # plt.title("Convolution")

        # # plt.figure()
        # # plt.imshow(wmap_patch_xcorr, cmap='gray')
        # # plt.title("Xcorr")

        # plt.figure()
        # plt.imshow(R, cmap='gray')
        # for x,y in max_set:
        #     plt.plot(x/d_factor, y/d_factor, 'bo', ms='6')
        # plt.title("Corrcoef")

        # plt.show()
        # print()

        # # q00_hist, q01_hist, q10_hist, q11_hist = patch_preprocess(patch_d, h_d, w_d)

        # # # # matching
        # # max_R = 0
        # # for y in range(m_d):
        # #     for x in range(n_d):
        # #         # print("(x,y) = ({},{})".format(x,y))
        # #         R = corrcoef_hists(y, x, h_d, w_d, wmap_d_hists, q00_hist, q01_hist, q10_hist, q11_hist)
        # #         # print("Corr coef R =", R)
        # #         if R > max_R:
        # #             max_R = R
        # #             max_x_d = x
        # #             max_y_d = y

        # # # upscaling x,y
        # # max_x = max_x_d * d_factor
        # # max_y = max_y_d * d_factor
        # # print("(x,y) = ({},{})".format(max_x,max_y))
        # # print("Corr coef R =", max_R)

        # # if not all([x,y] == sol_xy[p,:]):
        # #     print("Wrong solution. Got {}, should be {}".format([x,y], sol_xy[p,:]))
        

        # ##### BRAINSTORMING -------------------------------------------------------------------------------------------------------------

        # # x = int(sol_xy[p, 0])
        # # y = int(sol_xy[p, 1])

        # # print("(x,y):({},{})".format(x,y))

       
        # # print("Comparing on solution patch:")
        # # print("Comparing the whole image")
        # # compare(wmap, y, x, h, w, patch, True)
        # # # subdivide the patch into 4 quadrants:
        # # # Q00
        # # # Q01
        # # # Q10
        # # # Q11
        # # q00 = patch[:h//2, :w//2]
        # # q01 = patch[:h//2, w//2:]
        # # q10 = patch[h//2:, :w//2]
        # # q11 = patch[h//2:, w//2:]
        # # print("Comparing Q00")
        # # compare(wmap, y     ,      x, h//2, w//2, q00, True)
        # # print("Comparing Q01")
        # # compare(wmap, y     , x+w//2, h//2, w//2, q01, True)
        # # print("Comparing Q10")
        # # compare(wmap, y+h//2, x     , h//2, w//2, q10, True)
        # # print("Comparing Q11")
        # # compare(wmap, y+h//2, x+w//2, h//2, w//2, q11, True)

        # # print()
        # # print("Comparing on patch (0, 0):")
        # # x, y = 0, 0
        # # print("Comparing the whole image")
        # # compare(wmap, y, x, h, w, patch, False)
        # # # subdivide the patch into 4 quadrants:
        # # # Q00
        # # # Q01
        # # # Q10
        # # # Q11
        # # q00 = patch[:h//2, :w//2]
        # # q01 = patch[:h//2, w//2:]
        # # q10 = patch[h//2:, :w//2]
        # # q11 = patch[h//2:, w//2:]
        # # print("Comparing Q00")
        # # compare(wmap, y     ,      x, h//2, w//2, q00, False)
        # # print("Comparing Q01")
        # # compare(wmap, y     , x+w//2, h//2, w//2, q01, False)
        # # print("Comparing Q10")
        # # compare(wmap, y+h//2, x     , h//2, w//2, q10, False)
        # # print("Comparing Q11")
        # # compare(wmap, y+h//2, x+w//2, h//2, w//2, q11, False)

        # print()

                