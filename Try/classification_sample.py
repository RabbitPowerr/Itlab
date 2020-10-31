import sys
import os
import cv2
from argparse import ArgumentParser, SUPPRESS
import numpy as np
import logging as log
import InferenceEngineClassifier

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help = 'Path to an .xml \
    file with a trained model.', required = True, type = str)
    parser.add_argument('-w', '--weights', help = 'Path to an .bin file \
    with a trained weights.', required = True, type = str)
    parser.add_argument('-i', '--input', help = 'Path to \
    image file', required = True, type = str)
    parser.add_argument('-c', '--classes', help = 'File containing \
    classnames', type = str, default = None)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start IE classification sample")
    ie_classifier = InferenceEngineClassifier.InferenceEngineClassifier(configPath=args.model,
    weightsPath=args.weights)
    
    img = cv2.imread(args.input)
    


    prob = ie_classifier.classify(img)
    predictions = ie_classifier.get_top(prob, 5)
    predictions =predictions[::-1]
    names =["AAAAAA"]
    

    log.info("NAMES "+ names)
    
    log.info("Predictions: " + str(predictions))
    

    

if __name__ == '__main__':
 sys.exit(main())
