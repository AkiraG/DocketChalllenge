import tensorflow as tf
import numpy as np
import cv2 as cv
import re
import argparse
import pytesseract as ocr
import tqdm

import os
import warnings

warnings.filterwarnings('ignore')

TESSERACT_CONFIG = r'-l por --oem 3'
PATH_TO_SAVED_MODEL = "exported-models/my_model/saved_model"


def iterate_dir(path):
    images = [os.path.join(path, file) for file in os.listdir(path)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', file)]

    return images


def preprocessing_ocr(img):
    # TODO shring or enlarge automatically ( now only enlarges)
    width, height = img.shape[:2]
    multiplier = round(600 / width)
    #     result = cv2.medianBlur(img,5)
    result = cv.resize(img, None, fx=multiplier, fy=multiplier, interpolation=cv.INTER_CUBIC)
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    result = cv.GaussianBlur(result, (5, 5), 0)
    _, result = cv.threshold(result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #     result = cv2.adaptiveThreshold(result,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    kernel = np.ones((2, 2), np.uint8)
    result = cv.dilate(result, kernel, iterations=1)
    #     result = cv2.erode(result,kernel,iterations = 1)

    return result


def crop_roi(img, model):
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    crops = []
    height, width = img.shape[:2]
    for i in range(detections['detection_boxes'].shape[0]):
        if detections['detection_scores'][i] > 0.2:
            box = tuple(detections['detection_boxes'][i].tolist())
            ymin, xmin, ymax, xmax = box
            ymin = round(ymin * height)
            ymax = round(ymax * height)
            xmin = round(xmin * width)
            xmax = round(xmax * width)
            crops.append(img[ymin:ymax, xmin:xmax])

    return crops


def find_cpf(ocr_text_list):
    cpf_regex = re.compile('\d{3}\.\d{3}\.\d{3}\-\d{2}')
    for text in ocr_text_list:
        result = re.findall(cpf_regex, text.replace(' ', ''))
        if result:
            return result[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Busca por CPF em uma imagem',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i',
                        '--imageDir',
                        help='Diretório contendo as imagens',
                        type=str, default=os.getcwd())

    args = parser.parse_args()

    print('Loading model...', end='')
    model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print('Done!')

    images = iterate_dir(args.imageDir)

    for img in images:
        img_name = os.path.basename(img)
        loaded_img = cv.imread(img)
        roi_list = crop_roi(loaded_img, model)
        roi_preprocessed = [preprocessing_ocr(roi) for roi in roi_list]
        extracted_list = [ocr.image_to_string(roi, config=TESSERACT_CONFIG) for roi in roi_preprocessed]
        cpf = find_cpf(extracted_list)

        print(f'Image Name: {img_name} - CPF : {cpf}')


if __name__ == '__main__':
    main()
