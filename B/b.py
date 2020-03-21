"""
A Wise Person in the Mountain
    There used to be a very smart person who also thought they knew everything.
    As they grew older and got wiser,
    they realized they can only have some level of confidence in what they think the answer is.
    This person has moved to a mountain where they could focus on their studies,
    but every once in a while they would welcome pilgrims and answer their questions.


    In your college,
    you were given an assignment to evaluate the accuracy of the wise person's answers.
    Your task is to measure their ability to recognize correctness of statements posed as "Yes or No" questions.
    To this kind of question,
    wise person would always answer with a "Yes" and the confidence can be anything from 0% to 100%.
    For example,
    if wise person is certain the answer is "No",
    they would say their confidence in "Yes" is 0%.


    You've compiled a list of questions
    and for each question you've noted an answer which is widely considered to be correct.
    From the confidence provided by wise person, you are forming a prediction.

    Here is an example:
    Question: Is it true that the Earth is flat?
    Correct Answer: Yes (see task #3 of this assignment for more info)
    Wise person's confidence in "Yes": 5%
    Prediction: No (see below for explanation how a prediction is being formed)


    Your task is to aggregate all the answers into a set of measured quantities explained below.

Forming the prediction
    To evaluate the correctness of answers wise person has provided,
    you need to convert their confidence in "Yes" to a definite answer,
    which we will call prediction.
    
    If the confidence for "Yes" is larger or equal to some threshold T,
    the prediction will be "Yes",
    otherwise the prediction is a "No".
    Here are a few examples:

    Confidence in Yes   Threshold	Prediction
        90% 	            65%         Yes
        30% 	            50%         No
        10% 	            5%          Yes
        80% 	            95%         No

Measurement quantities
    Prediction types
        In the dataset you have:
            a number of questions whose correct answer is Yes. We call these Positive questions.
            a number of questions whose correct answer is No. We call these Negative questions.

        The types of predictions are summarized in the table:

                                Positive (P)	        Negative (N)
            Predicted positive	True positives (TP)	    False positives (FP)
            Predicted negative	False negatives (FN)	True negatives (TN)

        TPR (True Positive Rate) and FPR (False Positive Rate) can be computed as follows:
            TPR = TP / P
            FPR = FP / N

EER (Equal Error Rate)
    TPR and FPR depend on the threshold you chose.
    By varying the threshold (T) from 0% to 100% we get different values for TPR and FPR.
    There is a threshold T for which FPR and (1 - TPR) have the same value.
    This value is called EER.

Datasets
    As most of the students do, you've started working on this assignment a bit too late.
    Due to this, the correct answers and wise person's answers are all scattered around in different files on your computer.
    Each answer is in separate file:
        correct answers are in files named ca#.txt,
        and wise person's answers are in files named wpa#.txt
        (where # is a unique number of some question you had prepared).
    
    Unfortunately, in all this mess, you've lost some of the data,
    so not every correct answer has a matching wise person's answer, and vice versa.

Input
    Path to the input folder will be given through standard console input.
    If in doubt on how input is parsed, please refer to the solution of the first task (pixel task).

    Input folder may contain an arbitrarily deep subfolder tree which contains only files named as ca#.txt and wpa#.txt.

    Each ca#.txt contains only one word:
        Yes or No.
    Each wpa#.txt contains the confidence in "Yes" expressed in percents. For example:
        90%

    Note, wise person gives only integer confidences.

Output
    The results should be printed to standard output in the form of a comma separated string of values.

    These values are:

    (a) number of positive and (b) negative questions (this number does not depend on the availability of wise person's answers),
    (c) number of questions which can be used for evaluation (valid questions, they have both correct answer and prediction),
    (d) TPR and (e) FPR for 70% threshold,
    (f) Equal Error Rate.

    Example output:
        420,320,539,0.84,0.08,0.10
    
    Note that to compute TPR, FPR and EER, you can only use questions which have both correct answer and prediction.

Scoring
    *   Each of the correct numbers (a, b, c) will get you 8 points each,
    *   Each of the numbers (d, e) will get you 4 point each,
    *   number (f) with error less than 0.01 will get you 8 points.

    Hence, Each test case will bring you up to 40 points.
    Private set has 10 test cases for a total of 400 points for this task.


    All the values in the output are evaluated independently.
    If for some reason you cannot provide some value,
    just write nothing in between commas.
    For example, here is expected result and result to be evaluated:


                Positive	Negative	Valid       TPR     FPR	    EER
                questions   questions   questions   @ 70%   @ 70%
    Expected
    result	    1000        1000        1700        0.778	0.125	0.168
    Result	    1000        /           1650        0.78	0.08	0.16
    Scores	    2           0           0           1	    0	    2

    Corresponding output you'd write for the results from the above table is:
    1000,,1650,0.78,0.08,0.16

Constraints
    Time limit per test case is 5s.
    Memory limit per test case is 128MB.
"""

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

# printing -------------------------------------------------------------------
def print_data(ca_yes, ca_no, wpa, print_everything : bool = False) -> None:
    'Prints the extracted data'
    if print_everything:
        print("Matched correct answers == Yes")
        for i in ca_yes:
            if i in wpa:
                print(i, "Yes", wpa[i])
        print("Matched correct answers == No")
        for i in ca_no:
            if i in wpa:
                print(i, "No", wpa[i])


        print("Entries without wpa:")
        for i in ca_yes:
            if i not in wpa:
                print(i, "Yes", "???")
        for i in ca_no:
            if i not in wpa:
                print(i, "No", "???")

        print("Entries without ca:")
        for i in wpa:
            if (i not in ca_yes) and (i not in ca_no):
                print(i, "???", wpa[i])

    no_wpa_count = 0
    for i in ca_yes:
        if i not in wpa:
            no_wpa_count += 1
    for i in ca_no:
        if i not in wpa:
            no_wpa_count += 1

    no_ca_count = 0
    for i in wpa:
        if (i not in ca_yes) and (i not in ca_no):
            no_ca_count += 1


    n_ca = len(ca_yes)+len(ca_no)
    n_wpa = len(wpa)
    pct_missing = (100 * (no_wpa_count + no_ca_count)) // (n_ca + n_wpa)

    print("ca files number:", n_ca)
    print("wpa files number:", n_wpa)
    print("ca files without a wpa pair:", no_wpa_count)
    print("wpa files without a ca pair:", no_ca_count)
    print("Percentage of missing data:", str(pct_missing)+"%")



# main -------------------------------------------------------------------

# if __name__ == "__main__":    # TODO uncomment, indent bellow
# dir_path = input()    # TODO uncomment

paths = [   "B/public/set/1"
            ,"B/public/set/2"
            ,"B/public/set/3"
            ,"B/public/set/4"
            ,"B/public/set/5"
            ,"B/public/set/6"
            ,"B/public/set/7"
            ,"B/public/set/8"
            ,"B/public/set/9"
            ,"B/public/set/10"
        ]

for dir_path in paths:
    print()
    print("From dir:", dir_path)
    ca_yes, ca_no, wpa = get_data(dir_path)

    # print_data(ca_yes, ca_no, wpa)
    a, b, c, d, e, f = analyze_data(ca_yes, ca_no, wpa)
    print("{0},{1},{2},{3:.3f},{4:.3f},{5:.3f}".format(a, b, c, d, e, f))
