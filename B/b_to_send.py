import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

def is_correct_answer(filename : str) -> bool:
    'Checks if the given filename corresponds to a correct answer file'
    if filename is None:
        return None

    if filename[:2] == "ca" and filename[-4:] == ".txt" and filename[2:-4].isnumeric():
        return True
    else:
        return False

def is_wiseperson_answer(filename : str) -> bool:
    'Checks if the given filename corresponds to a correct answer file'
    if filename is None:
        return None

    if filename[:3] == "wpa" and filename[-4:] == ".txt" and filename[3:-4].isnumeric():
        return True
    else:
        return False

def number_correct_answer(filename : str) -> bool:
    'Checks if the given filename corresponds to a correct answer file'
    if filename is None:
        return None

    return int(filename[2:-4])

def number_wiseperson_answer(filename : str) -> bool:
    'Checks if the given filename corresponds to a correct answer file'
    if filename is None:
        return None
    
    return int(filename[3:-4])

def get_data(dir_path : str) -> (set, set, dict):
    'Extracts data from dir_path and all subfolders.'
    ca_yes = set()
    ca_no = set()
    wpa = {}

    for root, _, files in os.walk(dir_path):
        for filename in files:
            # print(filename)
            if filename.endswith(".txt"):
    
                if is_correct_answer(filename):
                    number = number_correct_answer(filename)
                    with open(os.path.join(root, filename)) as f:
                        answer = f.read()
                    if answer == "Yes":
                        ca_yes.add(number)
                    else:
                        ca_no.add(number)
    
                if is_wiseperson_answer(filename):
                    number = number_wiseperson_answer(filename)
                    with open(os.path.join(root, filename)) as f:
                        answer = f.read()
                        answer = int(answer[:-1])
                    wpa[number] = answer
    
    return ca_yes, ca_no, wpa


# def wpa_split(ca_yes : set, ca_not : set, wpa : dict) -> (list, list, int, int, int, int):
#     'Splits wpa into two sets based on the type of the ca'
#     n_for_yes = 0
#     n_for_yes_0 = 0
#     n_for_yes_100 = 0

#     n_for_no = 0
#     n_for_no_0 = 0
#     n_for_no_100 = 0
    
#     for i in ca_yes:
#         if i in wpa:
#             if 0 == wpa[i]:
#                 n_for_yes_0 += 1
#             elif 100 == wpa[i]:
#                 n_for_yes_100 += 1
#             else:
#                 n_for_yes += 1

#     for i in ca_no:
#         if i in wpa:
#             if 0 == wpa[i]:
#                 n_for_no_0 += 1
#             elif 100 == wpa[i]:
#                 n_for_no_100 += 1
#             else:
#                 n_for_no += 1
    
#     wpa_for_yes = [None for _ in range(n_for_yes)]
#     wpa_for_no = [None for _ in range(n_for_no)]
    
#     k = 0
#     for i in ca_yes:
#         if (i in wpa) and (0 != wpa[i]) and (100 != wpa[i]):
#             wpa_for_yes[k] = wpa[i]
#             k += 1
#     k = 0
#     for i in ca_no:
#         if (i in wpa) and (0 != wpa[i]) and (100 != wpa[i]):
#             wpa_for_no[k] = wpa[i]
#             k += 1
    
#     return wpa_for_yes, wpa_for_no, n_for_yes_0, n_for_no_0, n_for_yes_100, n_for_no_100

# def impute_data(ca_yes : set, ca_no : set, wpa : dict):
#     wpa_for_yes, wpa_for_no, n_for_yes_0, n_for_no_0, n_for_yes_100, n_for_no_100 = wpa_split(ca_yes, ca_no, wpa)
#     n_matched_yes = len(wpa_for_yes) + n_for_yes_0 + n_for_yes_100
#     n_matched_no = len(wpa_for_no) + n_for_no_0 + n_for_no_100
#     n_matched = n_matched_yes + n_matched_no

#     wpa_for_yes_arr = np.array(wpa_for_yes)
#     wpa_for_no_arr = np.array(wpa_for_no)

#     wpa_for_yes_mean = np.mean(wpa_for_yes_arr)
#     wpa_for_no_mean = np.mean(wpa_for_no_arr)

#     wpa_for_yes_std = np.std(wpa_for_yes_arr)
#     wpa_for_no_std = np.std(wpa_for_no_arr)
    
#     wpa_for_yes_median = np.median(wpa_for_yes_arr)
#     wpa_for_no_median = np.median(wpa_for_no_arr)
    
#     # wpa_for_yes_mode = stats.mode(wpa_for_yes_arr)
#     # wpa_for_no_mode = stats.mode(wpa_for_no_arr)
    
#     # wpa_for_yes_mode_frequency = wpa_for_yes_mode.count[0] / n_matched * 100
#     # wpa_for_no_mode_frequency = wpa_for_no_mode.count[0] / n_matched * 100

#     # plt.hist(wpa_for_yes_arr, bins = 100)
#     # print("wpa | ca == yes :")
#     # print("std ", wpa_for_yes_std)
#     # print("mean ", wpa_for_yes_mean)
#     # print("median ", wpa_for_yes_median)
#     # print("mode ", wpa_for_yes_mode, " frequency = ", wpa_for_yes_mode_frequency)
#     # print("n_for_yes_0 :", n_for_yes_0)
#     # print("n_for_yes_100 :", n_for_yes_100)
#     # plt.show()

#     # # # mapping wpa for anwers no to a log scale
#     # # wpa_for_no_arr = np.log(wpa_for_no_arr + 0.001)

#     # plt.hist(wpa_for_no_arr, bins = 100)
#     # print("wpa | ca == no :")
#     # print("std ", wpa_for_no_std)
#     # print("mean ", wpa_for_no_mean)
#     # print("median ", wpa_for_no_median)
#     # print("mode ", wpa_for_no_mode, " frequency = ", wpa_for_no_mode_frequency)
#     # print("n_for_no_0 :", n_for_no_0)
#     # print("n_for_no_100 :", n_for_no_100)
#     # plt.show()

#     # wpa_arr = np.array(wpa_for_yes + wpa_for_no)
#     # plt.hist(wpa_arr, bins = 100)
#     # plt.show()

#     # imputing ca values:
#     mu_y = wpa_for_yes_mean
#     mu_n = wpa_for_no_mean
#     std_y = wpa_for_yes_std
#     std_n = wpa_for_no_std
#     for i in wpa:
#         if (i not in ca_yes) and (i not in ca_no):
#             x = wpa[i]
#             if abs(x - mu_y)/std_y < abs(x - mu_n)/std_n:
#                 ca_yes.add(i)
#             else:
#                 ca_no.add(i)


#     n_unmatched_yes = 0
#     n_unmatched_no = 0
#     for i in ca_yes:
#         if i not in wpa:
#             n_unmatched_yes += 1

#     for i in ca_no:
#         if i not in wpa:
#             n_unmatched_no += 1

#     # calculating how much 0s and 100s to impute into wpa_yes
#     n_impute_yes_0 = round(n_for_yes_0 / n_matched_yes * n_unmatched_yes)
#     n_impute_yes_100 = round(n_for_yes_100 / n_matched_yes * n_unmatched_yes)

#     n_impute_no_0 = round(n_for_no_0 / n_matched_no * n_unmatched_no)
#     n_impute_no_100 = round(n_for_no_100 / n_matched_no * n_unmatched_no)

#     # imputing wpa values:
#     n0 = n_impute_yes_0
#     n100 = n_impute_yes_100
#     for i in ca_yes:
#         if i not in wpa:
#             if n0 > 0:
#                 wpa[i] = 0
#                 n0 -= 1
#             elif n100 > 0:
#                 wpa[i] = 100
#                 n100 -= 1
#             else:
#                 # wpa[i] = wpa_for_yes_mean
#                 wpa[i] = wpa_for_yes_median

#     n0 = n_impute_no_0
#     n100 = n_impute_no_100
#     for i in ca_no:
#         if i not in wpa:
#             if n0 > 0:
#                 wpa[i] = 0
#                 n0 -= 1
#             elif n100 > 0:
#                 wpa[i] = 100
#                 n100 -= 1
#             else:
#                 # wpa[i] = wpa_for_no_mean
#                 wpa[i] = wpa_for_no_median
    
#     return ca_yes, ca_no, wpa


def wpa_map(wpa : dict, T : int) -> (set, set):
    'Divides entries from wpa into two sets of predictions based on the given treshold'
    wpa_yes = set()
    wpa_no = set()
    for i, answer in wpa.items():
        if answer >= T:
            wpa_yes.add(i)
        else:
            wpa_no.add(i)
    
    return wpa_yes, wpa_no

 
def tpr_fpr(ca_yes : set, ca_no : set, wpa_yes : set, wpa_no : set) -> (float, float):
    'Returns TPR and FPR'
    TP, FP = 0, 0
    P, N = 0, 0
    
    for i in ca_yes:
        if i in wpa_yes:
            TP += 1
            P += 1
        if i in wpa_no:
            P += 1

    for i in ca_no:
        if i in wpa_yes:
            FP += 1
            N += 1
        if i in wpa_no:
            N += 1
    
    TPR = TP / P
    FPR = FP / N

    return TPR, FPR

# def eer_linear(ca_yes : set, ca_no : set, wpa : dict) -> float:
#     'Returns EER'

#     min_diff = 42
#     for T in range(0, 101):
#         wpa_yes, wpa_no = wpa_map(wpa, T)
#         tpr, fpr = tpr_fpr(ca_yes, ca_no, wpa_yes, wpa_no)

#         diff = abs(tpr + fpr - 1)
#         # print("For T={}, tpr, fpr = {}, {}, tpr+fpr = {}".format(T, tpr, fpr, tpr+fpr))

#         if diff <= min_diff:
#             min_diff = diff
#             eer_fpr = fpr
#         #     print("Less than or equal")
#         # else:
#         #     print("Greater than")

#     if min_diff <= 0.01:
#         eer = eer_fpr
#     else:
#         eer = 0

#     return eer

def eer_bs(ca_yes : set, ca_no : set, wpa : dict) -> float:
    'Returns EER'
    left = 0
    right = 100
    min_diff = 42
    while right - left > 1:
        T = (left + right)/2
        wpa_yes, wpa_no = wpa_map(wpa, T)
        tpr, fpr = tpr_fpr(ca_yes, ca_no, wpa_yes, wpa_no)
        diff = 1 - (tpr + fpr)

        if abs(diff) < min_diff:
            min_diff = abs(diff)
            eer_fpr = fpr

        if diff < 0:
            left = T
        elif diff > 0:
            right = T
        else:
            break

    return eer_fpr

def analyze_data(ca_yes : set, ca_no : set, wpa : dict):    #  -> (int, int, int, float, float, float):
    'Analyze the given data'

    no_ca_count = 0
    n_matched = 0
    for i in wpa:
        if (i not in ca_yes) and (i not in ca_no):
            no_ca_count += 1
        else:
            n_matched += 1

    a = len(ca_yes)
    b = len(ca_no)
    c = n_matched
    
    # # data imputing
    # ca_yes, ca_no, wpa = impute_data(ca_yes, ca_no, wpa)
    # for i in wpa:
    #     if (i not in ca_yes) and (i not in ca_no):
    #         print("Imputing into ca failed")
    
    # for i in ca_yes:
    #     if i not in wpa:
    #         print("Imputing into wpa failed")
    # for i in ca_no:
    #     if i not in wpa:
    #         print("Imputing into wpa failed")

    wpa_yes_70, wpa_no_70 = wpa_map(wpa, 70)
    d, e = tpr_fpr(ca_yes, ca_no, wpa_yes_70, wpa_no_70)

    f = eer_bs(ca_yes, ca_no, wpa)

    return a, b, c, d, e, f

# main -------------------------------------------------------------------
if __name__ == "__main__":
    dir_path = input()
    ca_yes, ca_no, wpa = get_data(dir_path)

    a, b, c, d, e, f = analyze_data(ca_yes, ca_no, wpa)
    print("{0},{1},{2},{3:.3f},{4:.3f},{5:.3f}".format(a, b, c, d, e, f))
