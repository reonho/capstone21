# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Simple class to run post processing of Detectnet-v2 Triton Inference outputs."""

import os

import numpy as np
from sklearn.cluster import DBSCAN as dbscan
from google.protobuf.text_format import Merge as merge_text_proto

from tao_triton.python.postprocessing.postprocessor import Postprocessor
import tao_triton.python.proto.postprocessor_config_pb2 as postprocessor_config_pb2
from tao_triton.python.types import KittiBbox
from tao_triton.python.postprocessing.utils import (
    denormalize_bounding_bboxes,
    iou_vectorized,
    pool_context,
    render_image,
    thresholded_indices,
    return_bbox_info
)
from tao_triton.python.utils.kitti import write_kitti_annotation
from PIL import Image

def load_clustering_config(config):
    """Load the clustering config."""
    proto = postprocessor_config_pb2.PostprocessingConfig()
    def _load_from_file(filename, pb2):
        if not os.path.exists(filename):
            raise IOError("Specfile not found at: {}".format(filename))
        with open(filename, "r") as f:
            merge_text_proto(f.read(), pb2)
    _load_from_file(config, proto)
    return proto
    

class DetectNetPostprocessor(Postprocessor):
    """Post processor for Triton outputs from a DetectNet_v2 client."""

    def __init__(self, batch_size, frames,
                 output_path, data_format, classes,
                 postprocessing_config, target_shape):
        """Initialize a post processor class for a classification model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
            classes (list): List of the class names.
            postprocessing_config (proto): Configuration elements of the dbscan postprocessor.
            target_shape (tuple): Shape of the model input.
        """
        self.pproc_config = load_clustering_config(postprocessing_config)
        self.classes = classes
        self.output_names = ["output_cov/Sigmoid",
                             "output_bbox/BiasAdd"]
        self.bbox_norm = [35., 35]
        self.offset = 0.5
        self.scale_h = 1
        self.scale_w = 1
        self.target_shape = target_shape
        self.stride = self.pproc_config.stride
        super().__init__(batch_size, frames, output_path, data_format)
        # Format the dbscan elements into classwise configurations for rendering.
        self.configure()

    def configure(self):
        """Configure the post processor object."""
        self.dbscan_elements = {}
        self.coverage_thresholds = {}
        self.box_color = {}
        classwise_clustering_config = self.pproc_config.classwise_clustering_config
        for class_name in self.classes:
            if class_name not in classwise_clustering_config.keys():
                raise KeyError("Cannot find class name {} in {}".format(
                    class_name, self.pproc_config.keys()
                ))
            self.dbscan_elements[class_name] = dbscan(
                eps=classwise_clustering_config[class_name].dbscan_config.dbscan_eps,
                min_samples=classwise_clustering_config[class_name].dbscan_config.dbscan_min_samples,
            )
            self.coverage_thresholds[class_name] = classwise_clustering_config[class_name].coverage_threshold
            self.box_color[class_name] = classwise_clustering_config[class_name].bbox_color

    def apply(self, results, this_id, render=True):
        """Apply the post processing to the outputs tensors.
        
        This function takes the raw output tensors from the detectnet_v2 model
        and performs the following steps:

        1. Denormalize the output bbox coordinates
        2. Threshold the coverage output to get the valid indices for the bboxes.
        3. Filter out the bboxes from the "output_bbox/BiasAdd" blob.
        4. Cluster the filterred boxes using DBSCAN.
        5. Render the outputs on images and save them to the output_path/images
        6. Serialize the output bboxes to KITTI Format label files in output_path/labels.
        """
        output_array = {}
        this_id = int(this_id)
        for output_name in self.output_names:
            output_array[output_name] = results.as_numpy(output_name).transpose(0, 1, 3, 2)
        assert len(self.classes) == output_array["output_cov/Sigmoid"].shape[1], (
            "Number of classes {} != number of dimensions in the output_cov/Sigmoid: {}".format(
                len(self.classes), output_array["output_cov/Sigmoid"].shape[1]
            )
        )
        abs_bbox = denormalize_bounding_bboxes(
            output_array["output_bbox/BiasAdd"], self.stride,
            self.offset, self.bbox_norm, len(self.classes), self.scale_w,
            self.scale_h, self.data_format, self.target_shape, self.frames,
            this_id - 1
        )
        valid_indices = thresholded_indices(
            output_array["output_cov/Sigmoid"], len(self.classes),
            self.classes,
            self.coverage_thresholds
        )
        batchwise_boxes = []
        for image_idx, indices in enumerate(valid_indices):
            covs = output_array["output_cov/Sigmoid"][image_idx, :, :, :]
            bboxes = abs_bbox[image_idx, :, :, :]
            imagewise_boxes = []
            for class_idx in range(len(self.classes)):
                clustered_boxes = []
                cw_config = self.pproc_config.classwise_clustering_config[
                    self.classes[class_idx]
                ]
                classwise_covs = covs[class_idx, :, :].flatten()
                classwise_covs = classwise_covs[indices[class_idx]]
                if classwise_covs.size == 0:
                    continue
                classwise_bboxes = bboxes[4*class_idx:4*class_idx+4, :, :]
                classwise_bboxes = classwise_bboxes.reshape(
                    classwise_bboxes.shape[:1] + (-1,)
                ).T[indices[class_idx]]
                pairwise_dist = \
                    1.0 * (1.0 - iou_vectorized(classwise_bboxes))
                labeling = self.dbscan_elements[self.classes[class_idx]].fit_predict(
                    X=pairwise_dist,
                    sample_weight=classwise_covs
                )
                labels = np.unique(labeling[labeling >= 0])
                for label in labels:
                    w = classwise_covs[labeling == label]
                    aggregated_w = np.sum(w)
                    w_norm = w / aggregated_w
                    n = len(w)
                    w_max = np.max(w)
                    w_min = np.min(w)
                    b = classwise_bboxes[labeling == label]
                    mean_bbox = np.sum((b.T*w_norm).T, axis=0)
                    mean_bbox = np.array(mean_bbox, dtype='float64')

                    # Compute coefficient of variation of the box coords
                    mean_box_w = mean_bbox[2] - mean_bbox[0]
                    mean_box_h = mean_bbox[3] - mean_bbox[1]
                    bbox_area = mean_box_w * mean_box_h
                    valid_box = aggregated_w > cw_config.dbscan_config.\
                        dbscan_confidence_threshold and mean_box_h > cw_config.minimum_bounding_box_height
                    if valid_box:
                        clustered_boxes.append(
                            KittiBbox(
                                self.classes[class_idx], 0, 0, 0,
                                mean_bbox, 0, 0, 0, 0,
                                0, 0, 0, confidence_score=aggregated_w
                            )
                        )
                    else:
                        continue
                imagewise_boxes.extend(clustered_boxes)
            batchwise_boxes.append(imagewise_boxes)

        if render:
            with pool_context(self.batch_size) as pool:
                batch_boxes_output = []
                for image_idx in range(self.batch_size):
                    current_idx = (this_id - 1) * self.batch_size + image_idx
                    if current_idx >= len(self.frames):
                        break
                    current_frame = self.frames[current_idx]
                    filename = os.path.basename(current_frame._image_path)
                    #Returns BBOX of all license plates in it
                    final_bboxes = return_bbox_info(current_frame, batchwise_boxes[image_idx])
                    batch_boxes_output.append([final_bboxes, filename])
                return batch_boxes_output

                    # output_label_file = os.path.join(
                    #     self.output_path, "infer_labels",
                    #     "{}.txt".format(os.path.splitext(filename)[0])
                    # )
                    # output_image_file = os.path.join(
                    #     self.output_path, "infer_images",
                    #     "{}.jpg".format(os.path.splitext(filename)[0])
                    # )
                    # if not os.path.exists(os.path.dirname(output_label_file)):
                    #     os.makedirs(os.path.dirname(output_label_file))
                    # if not os.path.exists(os.path.dirname(output_image_file)):
                    #     os.makedirs(os.path.dirname(output_image_file))
                    # processes.append(
                    #     pool.apply_async(
                    #         write_kitti_annotation, (output_label_file, batchwise_boxes[image_idx])
                    #     )
                    # )
                    # processes.append(
                    #     pool.apply_async(
                    #         render_image,
                    #         (current_frame, batchwise_boxes[image_idx],
                    #          output_image_file, self.box_color,
                    #          self.pproc_config.linewidth)
                    #     )
                    # )

                    # # Saving cropped image in output_path/cropeed_images
                    # # output_cropped_file = os.path.join(
                    # #     self.output_path, "cropped_images",
                    # #     "{}.jpg".format(os.path.splitext(filename)[0])
                    # # )

                    # # if not os.path.exists(os.path.dirname(output_cropped_file)):
                    # #     os.makedirs(os.path.dirname(output_cropped_file))

                    # # Crops Image
                    # # image = Image.open(current_frame._image_path)
                    # # image = image.crop(mean_bbox)
                    # # image.save(output_cropped_file, 'JPEG')

                    # processes.append(
                    #     pool.apply_async(
                    #         return_bbox,
                    #         (current_frame, batchwise_boxes[image_idx])
                    #     )
                    # )


                # for p in processes:
                #     p.wait()
