import os

import requests
from requests.exceptions import ConnectionError

from base_model_class import BaseModelClass
from bodyposenet_client import bodyposenet_predict
from tao_triton.python.postprocessing.utils import plot_keypoints


class BodyPoseNetClass(BaseModelClass):

    def __init__(self, client_info):
        '''
        Instantiate the classes with the information of the 
        querying party -- corresponding to a specific triton
        model.
        '''
        BaseModelClass.__init__(self, client_info)
        self._url = os.environ.get('API_URL')
        self._model_name = "bodyposenet"
        self._mode = "BodyPoseNet"

    def status(self):
        '''
        Returns the status of the model
        '''
        try:
            triton_server_url = "http://" + self._url + "/v2/health/ready"
            response = requests.get(triton_server_url)
        except ConnectionError as error:
            return {'HTTPStatus': 503, 'status': 'Inactive'}
        else:
            return {'HTTPStatus': 200, 'status': 'Active'}

    def predict(self, file_path, return_tensor=False):
        """Runs inference on images in file_path if it exists

        Args:
            file_path (string): File path of images to infer

        Returns:
            dict: If file path exists, returns dict containing results which stores the keypoints 
            detected for each image and skeleton edge names used for postprocessing.
            Else returns response indicating file path does not exist.
        """

        if os.path.exists(file_path):
            return self._predict(file_path, return_tensor=return_tensor)
        else:
            return {'HTTPStatus': 400,
                    'error': "File Path does not exist!"}

    def _predict(self, file_path, return_tensor=False):
        number_files = len([name for name in os.listdir(
            file_path) if os.path.isfile(file_path+name)])
        if number_files < 256:
            self._batch_size = 8
        else:
            self._batch_size = 16
        return bodyposenet_predict(model_name=self._model_name, mode=self._mode, url=self._url,
                                   image_filename=file_path, output_path='./', verbose=False, streaming=False, async_set=False,
                                   protocol='HTTP', model_version="", batch_size=self._batch_size, return_tensor=return_tensor)


if __name__ == '__main__':
    bp = BodyPoseNetClass(1)
    res = bp._predict('../input/bodyposenet')
    print(res)
    # image with keypoints/limbs rendered
    output = plot_keypoints(res, 'bp-sample.png',
                            '../input/bodyposenet/bp-sample.png')
    # TODO: Save the image in correct output path
