import logging as log
import cv2
import argparse
import ie_detector
import sys

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help = 'Path to an .xml \
    file with a trained model.', required = True, type = str)
    parser.add_argument('-w', '--weights', help = 'Path to an .bin file \
    with a trained weights.', required = True, type = str)
    parser.add_argument('-i', '--input', help = 'Path to \
    image file', required = True, type = str)
    parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
    (CPU)-targeted custom layers. Absolute path to a shared library \
    with the kernels implementation', type=str, default=None)
    parser.add_argument('-c', '--classes', help = 'File containing \
    classnames', type = str, default = None)
    return parser

def main():
    #args = build_argparser().parse_args()
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
    level=log.INFO, stream=sys.stdout)
    log.info("ok1")
    ie_detct = ie_detector.InferenceEngineDetector(configPath="F:\Sample\Detection\public\ssd300\FP16\ssd300.xml", weightsPath="F:\Sample\Detection\public\ssd300\FP16\ssd300.bin")
    img =cv2.imread("F:\Sample\aaa.jpg")
    detection= ie_detct.detect(img)
    log.info("ok1")
    
    #log.info(detection[0][0][0][1])
    image_detected = ie_detct.draw_detection(detection,img)
    cv2.imshow('Image with detections', image_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    sys.exit(main())
