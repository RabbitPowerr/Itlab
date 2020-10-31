import openvino.inference_engine 
import InferenceEngineClassifier
#
help(InferenceEngineClassifier)
i = openvino.inference_engine.ie_api.IECore()
a=1
b=2
print(a+b)