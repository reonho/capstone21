import argparse
from functools import partial
import logging
import os
import sys

from attrdict import AttrDict
import numpy as np
from tqdm import tqdm

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
from tao_triton.python.types import Frame, UserData
from tao_triton.python.postprocessing.bodyposenet_processor import BodyPoseNetPostprocessor
from tao_triton.python.model.bodyposenet_model import BodyPoseNetModel

logger = logging.getLogger(__name__)

TRITON_MODEL_DICT = {
    'bodyposenet': BodyPoseNetModel
}

POSTPROCESSOR_DICT = {
    'bodyposenet': BodyPoseNetPostprocessor
}


def completion_callback(user_data, result, error):
    """Callback function used for async_stream_infer()."""
    user_data._completed_requests.put((result, error))


def convert_http_metadata_config(_metadata, _config):
    """Convert to the http metadata to class Dict."""
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


def requestGenerator(batched_image_data, input_name, output_name, dtype, protocol,
                     num_classes=0):
    """Generator for triton inference requests.
    Args:
        batch_image_data (np.ndarray): Numpy array of a batch of images.
        input_name (str): Name of the input array
        output_name (list(str)): Name of the model outputs
        dtype: Tensor data type for Triton
        protocol (str): The protocol used to communicated between the Triton
            server and TAO Toolkit client.
        num_classes (int): The number of classes in the network.
    Yields:
        inputs
        outputs
        made_name (str): Name of the triton model
        model_version (int): Version number
    """
    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(
            out_name, class_count=num_classes
        ) for out_name in output_name
    ]

    yield inputs, outputs


def bodyposenet_predict(**FLAGS):
    """Sends image file path to client for inferences and returns postprocessed outputs

    Raises:
        Exception: If client creation fails
        InferenceSeverException: If failed to retrieve model config, metadata or if inference is unsuccessful.

    Returns:
        dict: Contains key results which stores the keypoints detected for each image and skeleton edge names used for postprocessing
    """

    log_level = "INFO"
    if FLAGS['verbose']:
        log_level = "DEBUG"
    # Configure logging to get Maglev log messages.
    logging.basicConfig(format='%(asctime)s [%(levelname)s] '
                               '%(name)s: %(message)s',
                        level=log_level)

    if FLAGS['streaming'] and FLAGS['protocol'].lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS['protocol'].lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS['url'], verbose=FLAGS['verbose'])
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            concurrency = 20 if FLAGS['async_set'] else 1
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS['url'], verbose=FLAGS['verbose'], concurrency=concurrency)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS['model_name'], model_version=FLAGS['model_version'])
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS['model_name'], model_version=FLAGS['model_version'])
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS['protocol'].lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config)

    triton_model = TRITON_MODEL_DICT[FLAGS['mode'].lower()].from_metadata(
        model_metadata, model_config)

    # Set target shape based on dimension format
    if triton_model.data_format == 1:  # NHWC:
        target_shape = (triton_model.h, triton_model.w, triton_model.c)
    elif triton_model.data_format == 2:  # NCHW
        target_shape = (triton_model.c, triton_model.h, triton_model.w)

    npdtype = triton_to_np_dtype(triton_model.triton_dtype)
    max_batch_size = triton_model.max_batch_size
    frames = []
    if os.path.isdir(FLAGS['image_filename']):
        frames = [
            Frame(os.path.join(FLAGS['image_filename'], f),
                  triton_model.data_format,
                  npdtype,
                  target_shape)
            for f in os.listdir(FLAGS['image_filename'])
            if os.path.isfile(os.path.join(FLAGS['image_filename'], f)) and
            os.path.splitext(f)[-1] in [".jpg", ".jpeg", ".png"]
        ]
    else:
        frames = [
            Frame(os.path.join(FLAGS['image_filename']),
                  triton_model.data_format,
                  npdtype,
                  target_shape)
        ]

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()
    args_postprocessor = [
        FLAGS['batch_size'], frames, FLAGS['output_path'], triton_model.data_format
    ]

    postprocessor = POSTPROCESSOR_DICT[FLAGS['mode'].lower()](
        *args_postprocessor)

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0

    if FLAGS['streaming']:
        triton_client.start_stream(partial(completion_callback, user_data))

    logger.info("Sending inference request for batches of data")
    with tqdm(total=len(frames)) as pbar:
        while not last_request:
            input_filenames = []
            repeated_image_data = []

            for idx in range(FLAGS['batch_size']):
                frame = frames[image_idx]
                img = frame.load_image()
                repeated_image_data.append(
                    triton_model.preprocess(
                        frame.as_numpy(img)
                    )
                )
                image_idx = (image_idx + 1) % len(frames)
                if image_idx == 0:
                    last_request = True

            if max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request
            try:
                req_gen_args = [batched_image_data, triton_model.input_names,
                                triton_model.output_names, triton_model.triton_dtype,
                                FLAGS['protocol'].lower()]
                req_gen_kwargs = {}
                req_generator = requestGenerator(
                    *req_gen_args, **req_gen_kwargs)
                for inputs, outputs in req_generator:
                    sent_count += 1
                    if FLAGS['streaming']:
                        triton_client.async_stream_infer(
                            FLAGS['model_name'],
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS['model_version'],
                            outputs=outputs)
                    elif FLAGS['async_set']:
                        if FLAGS['protocol'].lower() == "grpc":
                            triton_client.async_infer(
                                FLAGS['model_name'],
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=FLAGS['model_version'],
                                outputs=outputs)
                        else:
                            async_requests.append(
                                triton_client.async_infer(
                                    FLAGS['model_name'],
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=FLAGS['model_version'],
                                    outputs=outputs))
                    else:
                        responses.append(
                            triton_client.infer(FLAGS['model_name'],
                                                inputs,
                                                request_id=str(sent_count),
                                                model_version=FLAGS['model_version'],
                                                outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if FLAGS['streaming']:
                    triton_client.stop_stream()
                sys.exit(1)

            pbar.update(FLAGS['batch_size'])

    if FLAGS['streaming']:
        triton_client.stop_stream()

    if FLAGS['protocol'].lower() == "grpc":
        if FLAGS['streaming'] or FLAGS['async_set']:
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                processed_count += 1
                if error is not None:
                    print("inference failed: " + str(error))
                    sys.exit(1)
                responses.append(results)
    else:
        if FLAGS['async_set']:
            # Collect results from the ongoing async requests
            # for HTTP Async requests.
            for async_request in async_requests:
                responses.append(async_request.get_result())

    results = {}
    tensor_response = []
    logger.info(
        "Gathering responses from the server and post processing the inferenced outputs.")
    processed_request = 0
    with tqdm(total=len(frames)) as pbar:
        while processed_request < sent_count:
            response = responses[processed_request]
            if FLAGS['protocol'].lower() == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]

            batch_results = postprocessor.apply(
                response, this_id,
            )
            results = {**results, **{k: v.pop('results') for k,v in batch_results.items()}}
            
            tensor_response.append(batch_results)

            processed_request += 1
            pbar.update(FLAGS['batch_size'])
    logger.info("PASS")

    final_results = {}

    if FLAGS.get('return_tensor'):
        final_results['tensor_response'] = tensor_response
    final_results['results'] = results
    final_results['skeleton_edge_names'] = postprocessor.params['skeleton_edge_names']
    final_results['keypoints'] = postprocessor.params['keypoints']

    return final_results
