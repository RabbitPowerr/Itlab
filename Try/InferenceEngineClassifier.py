import cv2
import openvino.inference_engine.ie_api
import numpy as np
import logging as log

class InferenceEngineClassifier:
   def __init__(self, configPath = None, weightsPath = None):
      self.__core= openvino.inference_engine.ie_api.IECore()
      self.__net= self.__core.read_network(configPath,weightsPath)
      pass
   
   def get_top(self, prob, topN = 1):
      prob =prob[0]
      a= np.array([])
      for x in prob:
         tmp= max(x)[0]
         a= np.append(a,tmp)
      arr= np.argsort(a)
      return arr[-topN:]
      
   
   def _prepare_image(self, image, h, w):
      image = cv2.resize(image, (w, h))
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      blob = image.transpose((2, 0, 1))
      #blob = np.expand_dims(image, axis = 0)
      return blob
      
      
   def classify(self, image):
      input_blob = next(iter(self.__net.inputs))
      out_blob = next(iter(self.__net.outputs))
      n, c, h, w = self.__net.inputs[input_blob].shape
      blob =self._prepare_image(image,h,w)
      exec_net = self.__core.load_network(self.__net,  device_name = 'CPU')
      log.info("ok5\n")
      output = exec_net.infer(inputs = {input_blob: blob})

      log.info("ok6\n")
      output =output[out_blob]
      return output


   