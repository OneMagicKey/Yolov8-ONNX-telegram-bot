import os
import timeit
import unittest

import cv2
import numpy as np

from model.model import YoloOnnxDetection, YoloOnnxSegmentation


class YoloTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.detection_model = cls.load_detection_model()
            cls.segmentation_model = cls.load_segmentation_model()
            cls.img = cls.load_img()
            cls.img_empty = cls.create_empty_img()
        except:
            assert False

    @staticmethod
    def load_detection_model(path: str = '../checkpoints/detection/yolov5n.onnx'):
        return YoloOnnxDetection(path, input_size=(640, 640))

    @staticmethod
    def load_segmentation_model(path: str = '../checkpoints/segmentation/yolov8n-seg.onnx'):
        return YoloOnnxSegmentation(path, input_size=(640, 640))

    @staticmethod
    def load_img(path: str = '../images/zidane.jpg'):
        return cv2.imread(path, cv2.IMREAD_COLOR)

    @staticmethod
    def create_empty_img():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img.fill(255)

        return img

    def test_detection(self):
        img, yolo = self.img, self.detection_model
        output = yolo(img, raw=True)

        self.assertIsInstance(output[0], np.ndarray)

    def test_segmentation(self):
        img, yolo = self.img, self.segmentation_model

        output = yolo(img, raw=True)

        self.assertIsInstance(output[0], np.ndarray)
        self.assertEqual(type(output[0]), type(output[1]))

    def test_print_results(self):
        img, yolo = self.img, self.detection_model

        output = yolo(img)
        result_list = yolo.print_results(*output, language='en')

        self.assertIsInstance(result_list, list)

    def test_render_segmentation(self):
        img, yolo = self.img, self.segmentation_model
        path_save_to = "result_segmentation.jpg"

        classes, confs, boxes, masks = yolo(img)
        yolo.render(img, classes, confs, boxes, masks, save_path=path_save_to)

        assert os.path.exists(path_save_to)

    def test_render_detection(self):
        img, yolo = self.img, self.detection_model
        path_save_to = "result_detection.jpg"

        classes, confs, boxes = yolo(img)
        yolo.render(img, classes, confs, boxes, save_path=path_save_to)

        assert os.path.exists(path_save_to)

    def test_render_empty_segmentation(self):
        img, yolo = self.img_empty, self.segmentation_model
        path_save_to = "white_seg.jpg"

        classes, confs, boxes, masks = yolo(img)
        yolo.render(img, classes, confs, boxes, masks, save_path=path_save_to)
        result_list = yolo.print_results(classes, confs, boxes, masks)

        assert os.path.exists(path_save_to)
        self.assertEqual(len(result_list), 0)

    def test_render_empty_detection(self):
        img, yolo = self.img_empty, self.detection_model
        path_save_to = "white_det.jpg"

        classes, confs, boxes = yolo(img)
        yolo.render(img, classes, confs, boxes, save_path=path_save_to)
        result_list = yolo.print_results(classes, confs, boxes)

        assert os.path.exists(path_save_to)
        self.assertEqual(len(result_list), 0)


    # Performance
    # def test_many_detections(self, num_calls: int = 5):
    #     img, yolo = self.img, self.load_detection_model()
    #     first_time_inference = timeit.timeit(lambda: yolo(img), number=1)
    #     mean_time_inference = timeit.timeit(lambda: yolo(img), number=num_calls) / num_calls
    #
    #     print(f'{first_time_inference:.3f}s {mean_time_inference:.3f}s')
    #
    # def test_many_segmentation(self, num_calls: int = 5):
    #     img, yolo = self.img, self.load_segmentation_model()
    #     first_time_inference = timeit.timeit(lambda: yolo(img), number=1)
    #     mean_time_inference = timeit.timeit(lambda: yolo(img), number=num_calls) / num_calls
    #
    #     print(f'{first_time_inference:.3f}s {mean_time_inference:.3f}s')


if __name__ == '__main__':
    unittest.main()
