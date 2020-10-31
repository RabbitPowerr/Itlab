import openvino.inference_engine.ie_api 
import cv2

class InferenceEngineDetector:
    def __init__(self, configPath = None, weightsPath = None,
        extension=None):
        core= openvino.inference_engine.ie_api.IECore()
        self.__net= core.read_network(configPath,weightsPath)
        self.__exec_net = core.load_network(self.__net,  device_name = 'CPU')
        pass
    
    def _prepare_image(self,image,h,w):
        image=cv2.resize(image,(w,h))
        image= image.transpose((2,0,1))
        return image
    
    def detect(self,image):
        input_blob = next(iter(self.__net.inputs))
        out_blob = next(iter(self.__net.outputs))
        n,c,w,h=self.__net.inputs[input_blob].shape
        blob = self._prepare_image(image,h,w)
        output= self.__exec_net.infer(inputs={input_blob : blob})
        output = output[out_blob]
        return output
    
    def draw_detection(self,detections,image,cofidence=0.5,draw_text=True):
        sz = image.shape
        img =image
        ans = detections[0][0][0]
        for x in detections[0][0]:
            if(x[2]>ans[2]):
               ans=x
        point1=(int(ans[3]*sz[0]),int(ans[6]*sz[1]))
        point2=(int(ans[5]*sz[0]),int(ans[4]*sz[1]))
        color=(255,1,1)
        img= cv2.rectangle(img, point1,point2,color,thickness=1,lineType=None,shift=None)
               
        return img






