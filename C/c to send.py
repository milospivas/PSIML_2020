"""
    Task C - LABELING
"""

import json
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.animation as animation


def data_read(bboxes_path : str, joints_path : str) -> (list, list):
    'Reads and returns the bboxes and joints data from given paths'
    with open(bboxes_path) as f:
        bboxes_frames = json.load(f)
    
    with open(joints_path) as f:
        joints_frames = json.load(f)

    bboxes_frames = bboxes_frames["frames"]
    joints_frames = joints_frames["frames"]
    
    return bboxes_frames, joints_frames


def data_identify(bboxes_frames : list, joints_frames : list) -> (int, int, int, int, dict, dict):
    """ Returns the indices of the starting and stopping frame,
        the number of different bboxes and number of different joints,
        as well as index dictionaries that map identities of bboxes and joints
        to int values in range(0, n_bboxes), range(0, n_joints) respectively.
    """
    # finding the last frame_index
    # extracting bboxes identities
    starting_frame_idx = 1000000
    stopping_frame_idx = 0
    bboxes_index = {}
    bbox_idx = 0
    for frame in bboxes_frames:
        if frame["frame_index"] < starting_frame_idx:
            starting_frame_idx = frame["frame_index"]
        
        if frame["frame_index"] > stopping_frame_idx:
            stopping_frame_idx = frame["frame_index"]

        for bbox in frame["bounding_boxes"]:
            identity = bbox["identity"]
            if identity not in bboxes_index:
                bboxes_index[identity] = bbox_idx
                bbox_idx += 1
        
    # extracting joints identities
    joints_index = {}
    joint_idx = 0
    for frame in joints_frames:
        if frame["frame_index"] < starting_frame_idx:
            starting_frame_idx = frame["frame_index"]

        if frame["frame_index"] > stopping_frame_idx:
            stopping_frame_idx = frame["frame_index"]

        for joint in frame["joints"]:
            identity = joint["identity"]
            if identity not in joints_index:
                joints_index[identity] = joint_idx
                joint_idx += 1
    
    n_bboxes = len(bboxes_index)
    n_joints = len(joints_index)

    return starting_frame_idx, stopping_frame_idx, n_bboxes, n_joints, bboxes_index, joints_index


def data_extract_signals(   bboxes_frames : list,
                            joints_frames : list,
                            starting_frame_idx : int,
                            n_frames : int,
                            bboxes_index : dict,
                            joints_index : dict) -> (np.array,
                                                    np.array,
                                                    np.array,
                                                    np.array,
                                                    np.array,
                                                    np.array):
    'Extracts and returns x,y values for bboxes and joints and w,h values for bboxes'

    bboxes_x = np.zeros((n_frames, n_bboxes))   # since all values are normalized to [0,1], pure 0 will represent missing values - no signal
    bboxes_y = np.zeros((n_frames, n_bboxes))   # since all values are normalized to [0,1], pure 0 will represent missing values - no signal
    bboxes_w = np.zeros((n_frames, n_bboxes))   # since all values are normalized to [0,1], pure 0 will represent missing values - no signal
    bboxes_h = np.zeros((n_frames, n_bboxes))   # since all values are normalized to [0,1], pure 0 will represent missing values - no signal
    joints_x = np.zeros((n_frames, n_joints))   # since all values are normalized to [0,1], pure 0 will represent missing values - no signal
    joints_y = np.zeros((n_frames, n_joints))   # since all values are normalized to [0,1], pure 0 will represent missing values - no signal
    
    dval = 0.000000001
    for frame in bboxes_frames:
        frame_idx = frame["frame_index"] - starting_frame_idx
        for bbox in frame["bounding_boxes"]:
            bbox_idx = bboxes_index[bbox["identity"]]
            
            x = bbox["bounding_box"]["x"]
            y = bbox["bounding_box"]["y"]
            w = bbox["bounding_box"]["w"]
            h = bbox["bounding_box"]["h"]

            if x <= 0:
                x = dval
            if y <= 0:
                y = dval
            if x > 1:
                x = 1 
            if y > 1:
                y = 1 

            if w <= 0:
                w = dval
            if h <= 0:
                h = dval
            if w > 1:
                w = 1 
            if h > 1:
                h = 1 


            bboxes_x[frame_idx, bbox_idx] = x
            bboxes_y[frame_idx, bbox_idx] = y
            bboxes_w[frame_idx, bbox_idx] = w
            bboxes_h[frame_idx, bbox_idx] = h
    
    for frame in joints_frames:
        frame_idx = frame["frame_index"] - starting_frame_idx
        for joint in frame["joints"]:
            joint_idx = joints_index[joint["identity"]]

            x = joint["joint"]["x"]
            y = joint["joint"]["y"]

            if x <= 0:
                x = dval
            if y <= 0:
                y = dval
            if x > 1:
                x = 1 
            if y > 1:
                y = 1 
        

            joints_x[frame_idx, joint_idx] = x
            joints_y[frame_idx, joint_idx] = y

    return bboxes_x, bboxes_y, bboxes_w, bboxes_h, joints_x, joints_y


# def animate_frames(n_frames, n_bboxes, n_joints, bboxes_x, bboxes_y, bboxes_w, bboxes_h, joints_x, joints_y, bboxes_index, joints_index, starting_frame = 0, stopping_frame = None) -> None:
#     'Draw the bboxes and joints through frames'
#     if stopping_frame is None:
#         stopping_frame = n_frames

#     inch_width = 8
#     inch_height = 5
#     dpi = 100
#     width = inch_width * dpi
#     height = inch_height * dpi


#     fig = plt.figure(figsize=(inch_width, inch_height), dpi = dpi)
#     # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
#                         xlim=(0, width), ylim=(0, height))

#     joints, = ax.plot([], [], 'ro', ms=6)
#     bboxes = [patches.Rectangle((0, 0), 0, 0, fill=False) for _ in range(n_bboxes)]
    

#     joints_labels = [ax.text(0, 0, str(i), fontsize=12) for i in joints_index]
#     bboxes_labels = [ax.text(0, 0, str(i), fontsize=12) for i in bboxes_index]

#     def init():
#         # joints.set_data([], [])
#         for bbox in bboxes:
#             ax.add_patch(bbox)
#         return bboxes
#         # pass
    
#     def animate(i : int):
#         frame_idx = i + starting_frame
#         joints.set_data(joints_x[frame_idx,:] * width , joints_y[frame_idx,:] * height )
        
#         for joint, j in joints_index.items():
#             joints_labels[j].set_x(joints_x[frame_idx, j] * width)
#             joints_labels[j].set_y(joints_y[frame_idx, j] * height)
        
        
#         for bbox, j in bboxes_index.items():
#             w = bboxes_w[frame_idx, j] * width
#             h = bboxes_h[frame_idx, j] * height
#             x = bboxes_x[frame_idx, j] * width  #+ 0.5 * w 
#             y = bboxes_y[frame_idx, j] * height #+ 0.25 * h
#             bboxes[j].set_xy((x,y))
#             bboxes[j].set_width(w)
#             bboxes[j].set_height(h)

#             bboxes_labels[j].set_x(bboxes_x[frame_idx, j] * width)
#             bboxes_labels[j].set_y(bboxes_y[frame_idx, j] * height)
        
        
#         plt.title("Frame: "+str(frame_idx))
#         return joints, joints_labels, bboxes, bboxes_labels
    
#     frame_number = stopping_frame - starting_frame
#     ani = animation.FuncAnimation(fig, animate, init_func = init, frames = frame_number, interval = 1000)

#     plt.show()


def range_impute_missing(start_frame : int, stop_frame : int, identity : int, val : np.array) -> None:
    'Impute the missing data in the given range'

    # linear interpolation
    left_val = val[start_frame-1, identity]
    right_val = val[stop_frame, identity]

    n = stop_frame - start_frame
    delta_val = (right_val - left_val) / (n + 1)

    for i in range(n):
        val[start_frame + i, identity] = left_val + (i+1) * delta_val
    

def coordinate_impute_missing(n_frames : int, n_identities : int, val : np.array) -> (list, list):
    'Impute the missing data on given cooridnate'
    
    first_nonmissing = {}

    # find the first nonmissing elements
    for frame_idx in range(n_frames):
        if len(first_nonmissing) == n_identities:
            break

        for i in range(n_identities):
            if (i not in first_nonmissing) and (0 != val[frame_idx, i]):
                first_nonmissing[i] = frame_idx
    
    last_nonmissing = first_nonmissing.copy()

    for i in range(n_identities):
        if i not in first_nonmissing:
            first_nonmissing[i] = n_frames
            last_nonmissing[i] = 0

    for i in range(n_identities):
        for frame_idx in range(first_nonmissing[i]+1, n_frames):
            if 0 != val[frame_idx, i]:
                if last_nonmissing[i] < frame_idx - 1:
                    range_impute_missing(last_nonmissing[i]+1, frame_idx, i, val)
                last_nonmissing[i] = frame_idx
    
    first_nonmissing_list = [first_nonmissing[i] for i in range(n_identities)]
    last_nonmissing_list = [last_nonmissing[i] for i in range(n_identities)]
    # return first_nonmissing, last_nonmissing
 
    return first_nonmissing_list, last_nonmissing_list


def data_impute_missing(n_frames : int, n_bboxes : int, n_joints : int,
                        bboxes_x : np.array,
                        bboxes_y : np.array,
                        bboxes_w : np.array,
                        bboxes_h : np.array,
                        joints_x : np.array,
                        joints_y : np.array) -> (np.array, np.array, np.array, np.array):
    'Impute the missing data for bboxes and joints'

    first_nonmissing_bboxes_x, last_nonmissing_bboxes_x = coordinate_impute_missing(n_frames, n_bboxes, bboxes_x)
    first_nonmissing_bboxes_y, last_nonmissing_bboxes_y = coordinate_impute_missing(n_frames, n_bboxes, bboxes_y)
    first_nonmissing_bboxes_w, last_nonmissing_bboxes_w = coordinate_impute_missing(n_frames, n_bboxes, bboxes_w)
    first_nonmissing_bboxes_h, last_nonmissing_bboxes_h = coordinate_impute_missing(n_frames, n_bboxes, bboxes_h)
    first_nonmissing_joints_x, last_nonmissing_joints_x = coordinate_impute_missing(n_frames, n_joints, joints_x)
    first_nonmissing_joints_y, last_nonmissing_joints_y = coordinate_impute_missing(n_frames, n_joints, joints_y)
    
    first_nonmissing_bboxes = np.array([
                                        first_nonmissing_bboxes_x,
                                        first_nonmissing_bboxes_y,
                                        first_nonmissing_bboxes_w,
                                        first_nonmissing_bboxes_h]) 
    last_nonmissing_bboxes = np.array([
                                        last_nonmissing_bboxes_x,
                                        last_nonmissing_bboxes_y,
                                        last_nonmissing_bboxes_w,
                                        last_nonmissing_bboxes_h])

    first_nonmissing_joints = np.array([
                                        first_nonmissing_joints_x,
                                        first_nonmissing_joints_y]) 
    last_nonmissing_joints = np.array([
                                        last_nonmissing_joints_x,
                                        last_nonmissing_joints_y])

    first_nonmissing_bboxes = np.min(first_nonmissing_bboxes, 0)
    last_nonmissing_bboxes = np.max(last_nonmissing_bboxes, 0)
    first_nonmissing_joints = np.min(first_nonmissing_joints, 0)
    last_nonmissing_joints = np.max(last_nonmissing_joints, 0)

    return first_nonmissing_bboxes, last_nonmissing_bboxes, first_nonmissing_joints, last_nonmissing_joints


# def show_occurrence_intervals(first_nonmissing_bboxes, last_nonmissing_bboxes, first_nonmissing_joints, last_nonmissing_joints, bboxes_index, joints_index, n_frames):
#     'Plot the occurrence intervals of each bbox and joint'

#     b_x = np.transpose(np.array([first_nonmissing_bboxes, last_nonmissing_bboxes]))
#     j_x = np.transpose(np.array([first_nonmissing_joints, last_nonmissing_joints]))
    
    
#     b_y = np.transpose([np.array(list(range(len(bboxes_index))))] * 2)
#     j_y = np.transpose([np.array(list(range(len(joints_index))))] * 2) + len(bboxes_index)

#     fig = plt.figure()
#     # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = fig.add_subplot()

#     text_x_offset = 5
#     text_y_offset = 0.2
#     bboxes_color = 'black'
#     joints_color = 'red'
#     for b, idx in bboxes_index.items():
#         plt.plot(b_x[idx, :], b_y[idx, :], marker = 'o', color = bboxes_color)
#         plt.text(b_x[idx, 0] + text_x_offset, b_y[idx, 0] + text_y_offset, str(b), fontsize=12, color = bboxes_color)

#     for j, idx in joints_index.items():
#         plt.plot(j_x[idx, :], j_y[idx, :], marker = 'o', color = joints_color)
#         plt.text(j_x[idx, 0] + text_x_offset, j_y[idx, 0] + text_y_offset, str(j), fontsize=12, color = joints_color)

#     plt.title("Occurrence intervals of bboxes and joints")
#     plt.show()

def data_correlate( n_bboxes : int, n_joints : int,
                    bboxes_x : np.array,
                    bboxes_y : np.array,
                    bboxes_w : np.array,
                    bboxes_h : np.array,
                    joints_x : np.array,
                    joints_y : np.array,
                    first_nonmissing_bboxes : np.array,
                    last_nonmissing_bboxes : np.array,
                    first_nonmissing_joints : np.array,
                    last_nonmissing_joints : np.array
                ) -> np.array:
    'Compute correlation coefficients between each bbox and joint xy signals and combine xy values into single value'
    # joint position estimate
    joint_height_proportion = 1/4
    bboxes_j_x = bboxes_x + bboxes_w/2
    bboxes_j_y = bboxes_y + bboxes_h * joint_height_proportion


    F = np.zeros((n_bboxes, n_joints))

    dval = 0.000000001
    pos_error = 0.02
    ghosting_error = 0.15
    min_correlation_len = 4
    p = 0.4

    for b in range(n_bboxes):
        for j in range(n_joints):
            s = max(first_nonmissing_bboxes[b], first_nonmissing_joints[j])
            e = min(last_nonmissing_bboxes[b], last_nonmissing_joints[j])

            signal_len = e+1 - s
            if signal_len <= 0:
                Rx = -1 #0
                Ry = -1 #0
                Dy = 1
            else:
                
                # check if the neckjoint is inside the bounding box
                is_inside = True
                allowed_time_outside = signal_len * ghosting_error

                n_frames_outside = 0
                y_distance = 0
                for frame_idx in range(s, e+1):
                    j_x = joints_x[frame_idx, j]
                    j_y = joints_y[frame_idx, j]
                    # b_j_x = bboxes_j_x[frame_idx, b]
                    b_j_y = bboxes_j_y[frame_idx, b]
                    b_x = bboxes_x[frame_idx, b]
                    b_y = bboxes_y[frame_idx, b]
                    b_w = bboxes_w[frame_idx, b]
                    b_h = bboxes_h[frame_idx, b]

                    deltaY = abs(b_j_y - j_y)
                    y_distance += deltaY

                    if  (j_x < (b_x - pos_error)) or (j_y < (b_y - pos_error)) or ((b_x + b_w + pos_error) < j_x) or ((b_y + b_h + pos_error) < j_y):
                        is_inside = False
                        n_frames_outside += 1
                
                Dy = y_distance/signal_len

                if is_inside and signal_len < min_correlation_len:
                    Rx = 0.9
                    Ry = 0.9
                elif n_frames_outside < allowed_time_outside:
                    bboxes_x[s, b] += dval
                    bboxes_y[s, b] += dval
                    joints_x[s, j] += dval
                    joints_y[s, j] += dval
                    Rx = np.corrcoef(bboxes_j_x[s:e+1, b], joints_x[s:e+1, j], rowvar = False)
                    Ry = np.corrcoef(bboxes_j_y[s:e+1, b], joints_y[s:e+1, j], rowvar = False)
                    
                    Rx = Rx[0,1]
                    Ry = Ry[0,1]

                    if np.isnan(Rx):
                        Rx = 0

                    if np.isnan(Ry):
                        Ry = 0
                else:
                    Rx = -1 #0
                    Ry = -1 #0

            # normalizing R to [0, 1]
            Px = (Rx + 1) / 2
            Py = (Ry + 1) / 2
            A = 1 - Dy

            if 0 != Py + A:
                PyA = 2 * (Py * A) / (Py + A)
                F[b, j] = PyA
    
    return F

# def print_F(n_bboxes, n_joints, F, index_bboxes, index_joints):
#     'prints F matrix'
#     for b in range(n_bboxes):
#         print("{0:2s}".format(index_bboxes[b]), end=": ")
#         for j in range(n_joints):
#             if 0 != F[b,j]:
#                 print("{0:.3f}".format(F[b,j]), end="  ")
#             else:
#                 print("{0:>4}".format("0"), end="   ")
                
#         print()
#     print("  ", end="")
#     for j in range(n_joints):
#         print("{0:>6}".format(index_joints[j]), end=" ")
#     print()
    

def recursive_cover(F : np.array, S : list, skip : set, chosen : set, j : int, m : int) -> (int, dict):
    'Backtracking, covering all solutions'
    if j in skip:
        return recursive_cover(F, S, skip, chosen, j+1, m)
    
    max_val = 0
    max_b_dict = {}
    
    if j < m:
        for b in S[j]:
            if b not in chosen:
                chosen.add(b)
                val, b_dict = recursive_cover(F, S, skip, chosen, j+1, m)
                chosen.remove(b)

                if len(b_dict) < m - (j+1) - len(skip):
                    continue
                val += F[b, j]
                b_dict[j] = b
                if val > max_val:
                    max_val = val
                    max_b_dict = b_dict
    
    return max_val, max_b_dict

def find_match(n : int, m : int, F_val : np.array, axis : int = 0) -> dict:
    '''Matches indices of F_val\'s columns and rows a la Non-attacking Rooks backtracking
    such that the sum of matched elements is maximised.
    '''
    if axis != 0:
        temp = n
        n = m
        m = temp
        F = np.transpose(F_val)
    else:
        F = F_val
        
    S = [set() for _ in range(m)]
    skip = set()
    for j in range(m):
        for b in range(n):
            if F[b, j] != 0:
                S[j].add(b)
        if len(S[j]) == 0:
            skip.add(j)

    chosen = set()

    max_sum, index_map = recursive_cover(F, S, skip, chosen, 0, m)

    return index_map

def data_match(n_bboxes : int, n_joints : int, bboxes_index : dict, joints_index : dict, F : np.array):
    'Match joints to bboxes given correlation coefficients'

    joint_bbox_map = {}

    index_bboxes = {idx : bbox for bbox, idx in bboxes_index.items()}
    index_joints = {idx : joint for joint, idx in joints_index.items()}

    # print_F(n_bboxes, n_joints, F, index_bboxes, index_joints)
    
    if n_bboxes > n_joints:
        bboxes_matches = find_match(n_bboxes, n_joints, F)

        for joint_idx, bbox_idx in bboxes_matches.items():
            joint_bbox_map[index_joints[joint_idx]] = index_bboxes[bbox_idx]
    else:
        joints_matches = find_match(n_bboxes, n_joints, F, axis=1)

        for bbox_idx, joint_idx in joints_matches.items():
            joint_bbox_map[index_joints[joint_idx]] = index_bboxes[bbox_idx]
    
    return joint_bbox_map

# def solution_map(solution_path : str):
#     jb_map = {}
#     with open(solution_path) as f:
#         lines = [line.rstrip() for line in f]
#         for s in lines:
#             j, b = s.split(":")
#             jb_map[j] = b
#     return jb_map

# main -------------------------------------------------------------------
if __name__ == "__main__":
    bboxes_path = input()
    joints_path = input()

    bboxes_frames, joints_frames = data_read(bboxes_path, joints_path)
    
    starting_frame_idx, stopping_frame_idx, n_bboxes, n_joints, bboxes_index, joints_index = data_identify(bboxes_frames, joints_frames)
    n_frames = stopping_frame_idx - starting_frame_idx + 1
   
    bboxes_x, bboxes_y, bboxes_w, bboxes_h, joints_x, joints_y = data_extract_signals(bboxes_frames, joints_frames, starting_frame_idx, n_frames, bboxes_index, joints_index)

    first_nonmissing_bboxes, last_nonmissing_bboxes, first_nonmissing_joints, last_nonmissing_joints = data_impute_missing(n_frames, n_bboxes, n_joints, bboxes_x, bboxes_y, bboxes_w, bboxes_h, joints_x, joints_y)

    F = data_correlate(n_bboxes, n_joints, bboxes_x, bboxes_y, bboxes_w, bboxes_h, joints_x, joints_y, first_nonmissing_bboxes, last_nonmissing_bboxes, first_nonmissing_joints, last_nonmissing_joints)

    joint_bbox_map = data_match(n_bboxes, n_joints, bboxes_index, joints_index, F)

    for j, b in joint_bbox_map.items():
        print(j+":"+b)
