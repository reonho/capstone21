import os
from .base_model_class import BaseModelClass
from .lpd_client import lpd_predict

class LpdModelClass(BaseModelClass):

    def __init__(self, client_info):
        '''
        Instantiate the classes with the information of the 
        querying party -- corresponding to a specific triton
        model.
        '''
        BaseModelClass.__init__(self, client_info)
        self._post_processing_config = "/app/triton_client/model_client/tao_triton/python/clustering_specs/clustering_config_lpdnet.prototxt"
        self._url = "35.240.147.255:6000"
        self._model_name = "lpdnet_usa"
        self._mode = "DetectNet_v2"
        self._class_list = "license_plate"
    
    def status(self):
        '''
        Returns the status of the model
        '''
        return {'status':'active'}

    
    def predict(self, file_path):
        '''
        Returns the output of the model after inference.
        '''
        if os.path.exists(file_path):
            return self._predict(file_path)
        else:
            return {'HTTPStatus':400, 
                    'error':"File Path does not exist!"}

    def _predict(self, file_path):
        number_files = len([name for name in os.listdir(file_path) if os.path.isfile(file_path+name)])
        if number_files < 256:
            self._batch_size = 8
        else:
            self._batch_size = 32
        return lpd_predict(model_name = self._model_name, mode = self._mode, class_list = self._class_list, \
                            output_path = "./output/lpdnet_usa", postprocessing_config = self._post_processing_config, \
                            url = self._url, image_filename = file_path, verbose = False, streaming = False, async_set = False, \
                            protocol = 'HTTP', model_version = "", batch_size = self._batch_size)

# test_model = LpdModelClass("hellosss");
# print(test_model.predict("../input/"))
# # test_model.predict("../input/")