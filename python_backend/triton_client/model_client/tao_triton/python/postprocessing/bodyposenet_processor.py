
import os
import math

import numpy as np
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter

from tao_triton.python.postprocessing.postprocessor import Postprocessor


class BodyPoseNetPostprocessor(Postprocessor):

    def __init__(self, batch_size, frames,
                 output_path, data_format,):
        """Initialize a post processor class for BodyPoseNet.

        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        self.output_names = ["heatmap_out/BiasAdd:0",
                             "conv2d_transpose_1/BiasAdd:0"]

        self.params = dict(
            stride=8,
            paf_gaussian_sigma=0.03,
            heatmap_gaussian_sigma=7.0,
            heatmap_threshold=0.1,
            paf_threshold=0.05,
            paf_ortho_dist_thresh=1,
            min_keypoints=4,
            min_avg_score=0.4,
            keypoints=[
                "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
                "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
                "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
                "left_eye", "right_ear", "left_ear"],
            # find connection in the specified sequence, center 29 is in the position 15
            limbSeq=[[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                     [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                     [1, 16], [16, 18], [3, 17], [6, 18]],
            # the middle joints heatmap correpondence
            mapIdx=[[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                    [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                    [55, 56], [37, 38], [45, 46]],
            skeleton_edge_names=[
                ["neck", "right_hip"], ["right_hip", "right_knee"], [
                    "right_knee", "right_ankle"],
                ["neck", "left_hip"], ["left_hip", "left_knee"], [
                    "left_knee", "left_ankle"],
                ["neck", "right_shoulder"], ["right_shoulder",
                                             "right_elbow"], ["right_elbow", "right_wrist"],
                ["right_shoulder", "right_ear"], ["neck", "left_shoulder"], [
                    "left_shoulder", "left_elbow"],
                ["left_elbow", "left_wrist"], [
                    "left_shoulder", "left_ear"], ["neck", "nose"],
                ["nose", "right_eye"], ["nose", "left_eye"], [
                    "right_eye", "right_ear"],
                ["left_eye", "left_ear"]
            ]
        )

        super().__init__(batch_size, frames, output_path, data_format)

        self.last_batch_id = (len(self.frames) // self.batch_size) + 1
        self.last_batch_size = len(self.frames) % self.batch_size

    def apply(self, results, this_id):
        """Postprocesses the tensor outputs to identify the keypoints for each person detected

        1. Resizes heatmap and part affinity maps
        2. Identifies peaks using gaussian filter on heatmap
        3. Identifies connections using part affinity maps and limb sequence
        4. Creates list of coordinates of keypoints and a subset indicating the keypoints for each person

        Args:
            results (InferResult): Batch results from Triton Inference Server
            this_id (int): Index of batch

        Returns:
            dict: Dictionary where key is image filename, value is a list for each person which contains the coordinates of each keypoint
        """

        output_array = {}
        this_id = int(this_id)
        for output_name in self.output_names:
            output_array[output_name] = results.as_numpy(output_name)

        batch_results = {}
        curr_batch_size = self.batch_size if this_id != self.last_batch_id else self.last_batch_size
        for i in range(curr_batch_size):  # process each image in the batch
            output = {}
            frame_id = (this_id-1)*self.batch_size + i
            frame = self.frames[frame_id]
            filename = os.path.basename(frame._image_path)

            heatmap, paf = self._resize_outputs(output_array, frame, i)
            all_peaks, peak_counter = self._find_peaks(heatmap)
            connection_all, special_k = self._find_connections(
                all_peaks, paf, frame)
            subset, candidate = self._find_subset(
                all_peaks, connection_all, special_k)
            results = []
            for person in subset:
                temp = {}
                # for each part found
                for i in range(len(self.params['keypoints'])):
                    idx = int(person[i])
                    if idx != -1:
                        coords = candidate[idx][:2].astype('float64')
                        temp[self.params['keypoints'][i]] = coords

                # score for overall configuration
                temp['score'] = person[-2].astype('float64')
                temp['total'] = int(person[-1])  # total parts found
                
                results.append(temp)

            batch_results[filename] = {'results':results, 'paf':paf, 'heatmap':heatmap }

        return batch_results

    def _resize_outputs(self, output_array, frame, index):
        """Resizes heatmap and part affinity maps to original image dimension

        Args:
            output_array (InferResult): Raw tensor outputs from BodyPoseNet
            frame (tao_triton.python.types.Frame): Frame object for the input image
            index (int): Index of the image in the current batch

        Returns:
            heatmap (np.ndarray): Resized heatmap
            paf (np.ndarray): Resized part affinity map

        """
        orig_width = frame.width
        orig_height = frame.height

        heatmap = output_array['heatmap_out/BiasAdd:0'][index]
        heatmap = cv.resize(
            heatmap, (0, 0), fx=self.params['stride'], fy=self.params['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = cv.resize(heatmap, (orig_width, orig_height),
                            interpolation=cv.INTER_CUBIC)

        paf = output_array['conv2d_transpose_1/BiasAdd:0'][index]
        paf = cv.resize(
            paf, (0, 0), fx=self.params['stride'], fy=self.params['stride'], interpolation=cv.INTER_CUBIC)
        paf = cv.resize(paf, (orig_width, orig_height),
                        interpolation=cv.INTER_CUBIC)

        return heatmap, paf

    def _find_peaks(self, heatmap):
        """Find peaks in heatmap

        Args:
            heatmap (numpy.ndarray): Resized heatmap array, with shape (image_width, image_height, number of keypoints+1)

        Returns:
            all_peaks (np.ndarray): List of peaks i.e. (x_coord, y_coord, score, peak_id)
            peak_counter (int): Total number of peaks found
        """
        all_peaks = []
        peak_counter = 0
        for part in range(len(self.params['keypoints'])):
            map_ori = heatmap[:, :, part]
            map_gaus = gaussian_filter(
                map_ori, sigma=self.params['heatmap_gaussian_sigma'])

            map_left = np.zeros(map_gaus.shape)
            map_left[1:, :] = map_gaus[:-1, :]
            map_right = np.zeros(map_gaus.shape)
            map_right[:-1, :] = map_gaus[1:, :]
            map_up = np.zeros(map_gaus.shape)
            map_up[:, 1:] = map_gaus[:, :-1]
            map_down = np.zeros(map_gaus.shape)
            map_down[:, :-1] = map_gaus[:, 1:]

            peaks_binary = np.logical_and.reduce((map_gaus >= map_left, map_gaus >= map_right,
                                                  map_gaus >= map_up, map_gaus >= map_down, map_gaus > self.params['heatmap_threshold']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(
                peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [
                peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return all_peaks, peak_counter

    def _find_connections(self, all_peaks, paf, frame):
        """Identifies connections from peaks and part affinity map

        Args:
            all_peaks (np.ndarray): List of peaks i.e. (x_coord, y_coord, score, peak_id)
            paf (np.ndarray): Resized part affinity map
            frame (tao_triton.python.types.Frame): Frame object for the input image

        Returns:
            connection_all (np.ndarray): Array containing connections
            special_k (np.ndarray): List of map indexes to ignore
        """
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(self.params['mapIdx'])):
            score_mid = paf[:, :, [x-19 for x in self.params['mapIdx'][k]]]
            candA = all_peaks[self.params['limbSeq'][k][0]-1]
            candB = all_peaks[self.params['limbSeq'][k][1]-1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = self.params['limbSeq'][k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        if np.all(vec == [0, 0]):
                            continue
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(
                            vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(
                            score_midpts)/len(score_midpts) + min(0.5*frame.width/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > 0.05)[
                                         0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0

                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

                connection_candidate = sorted(
                    connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if(i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack(
                            [connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return connection_all, special_k

    def _find_subset(self, all_peaks, connection_all, special_k):
        """[summary]

        Args:
            all_peaks (np.ndarray): List of peaks i.e. (x_coord, y_coord, score, peak_id)
            connection_all (np.ndarray): Array containing connections
            special_k (np.ndarray): List of map indexes to ignore

        Returns:
            subset (numpy.ndarray): Array where each item is a list of keypoints for each person detected.
            For each list, the 2nd last element is the configuration score, the last element is the number of keypoints detected for that person.
            The remaining elements are the indices of the keypoints contained in candidate.
            candidate (numpy.ndarray): List of keypoint candidates
        """
        subset = -1 * np.ones((0, 20))
        candidate = np.array(
            [item for sublist in all_peaks for item in sublist])

        for k in range(len(self.params['limbSeq'])):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.params['limbSeq'][k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(
                                int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(
                            int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(
                                int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,
                                                                  :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # Removes subsets based on configuration score or number of keypoints
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < self.params['min_keypoints'] or subset[i][-2]/subset[i][-1] < self.params['min_avg_score']:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        return subset, candidate
