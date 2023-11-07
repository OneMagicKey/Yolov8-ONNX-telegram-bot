from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple

import cv2
import numpy as np

from model.plots import COCO_names_en, COCO_names_ru, Colors
from model.utils import (
    draw_box,
    draw_masks,
    letterbox,
    process_masks,
    xywh2box,
    xywh2xyxy,
)


class YoloOnnx(ABC):
    def __init__(
        self,
        checkpoint: str,
        input_size: tuple[int, int] = (640, 640),
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.input_size = input_size  # (h, w)
        self.conf = conf
        self.iou = iou
        self.version = int(checkpoint.split("/yolov")[1][0])
        self.model = self.build_model(checkpoint)

        self.colors = Colors()
        self.labels_name = defaultdict(
            lambda: COCO_names_en, en=COCO_names_en, ru=COCO_names_ru
        )

    def build_model(self, checkpoint: str) -> cv2.dnn.Net:
        model = cv2.dnn.readNetFromONNX(checkpoint)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return model

    def forward_pass(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray]:
        """
        Create a blob from an image and propagate it through the model.

        :param img: np.array of shape (h, w, 3)
        :return: (boxes_array) or (boxes_and_masks_array, protos_array)
        """
        (w, h) = self.input_size[::-1]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (w, h), swapRB=True, crop=False)  # bs, c, h, w
        self.model.setInput(blob)

        out_blob_names = self.model.getUnconnectedOutLayersNames()
        output = self.model.forward(outBlobNames=out_blob_names)

        return output

    def print_results(
        self,
        classes: np.ndarray,
        confs: np.ndarray,
        boxes: np.ndarray,
        masks: None | np.ndarray = None,
        language: str = "en",
    ) -> list:
        BoxInfo = namedtuple("BoxInfo", ["class_id", "class_name", "conf", "box"])
        result_list = [
            BoxInfo(class_id, self.labels_name[language][class_id], conf, box)
            for class_id, conf, box in zip(classes, confs, boxes)
        ]
        return result_list

    def version_handler(self, output: np.ndarray, nc: int) -> tuple[np.ndarray, int]:
        """
        Convert output to the predefined format.

        :param output: an array with boxes from the model's output
        :param nc: number of classes
        :return: output with shape (num_boxes, probs_start_idx+num_classes+num_masks) and probs_start_idx
        """
        if self.version == 8:
            probs_start_idx = 4
            output = output.transpose((0, 2, 1)).squeeze(0)

        else:  # v5
            probs_start_idx = 5
            output = output[output[..., 4] > self.conf]
            output[..., probs_start_idx : probs_start_idx + nc] *= output[..., 4:5]  # conf = obj_conf * cls_conf

        return output, probs_start_idx

    def nms(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        return_masks: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Perform non-maximum suppression on the given boxes array.
        """
        nc = len(self.labels_name["en"])
        output, probs_id = self.version_handler(output, nc)

        classes = output[..., probs_id : probs_id + nc].argmax(axis=-1)
        boxes = xywh2box(output[..., :4], ratio, padw=pad[0], padh=pad[1])
        confs = output[..., probs_id : probs_id + nc]
        confs = confs[np.arange(classes.shape[-1]), classes]

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.iou)
        indices = list(indices)

        if return_masks:
            masks = output[indices, probs_id + nc :]
            return classes[indices], confs[indices], xywh2xyxy(boxes[indices]), masks

        return classes[indices], confs[indices], xywh2xyxy(boxes[indices])

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, img: np.ndarray, raw: bool = False) -> tuple[np.ndarray, ...]:
        pass


class YoloOnnxDetection(YoloOnnx):
    def render(
        self,
        img: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        boxes: np.ndarray,
        save_path: str | None = None,
        hide_conf: bool = True,
        language: str = "en",
    ) -> np.ndarray:
        """
        Render image with provided boxes and classes.

        :param img: input image (h, w, 3)
        :param classes: the array of class_ids with shape (n, ), n is number of boxes after nms
        :param confs: the array of confs with shape (n, )
        :param boxes: the array of boxes with shape (n, 4)
        :param save_path: save/to/file.jpg
        :param hide_conf: hide confidence score from the labels
        :param language: label language
        :return: rendered image
        """
        result_img = img.copy()
        for class_id, conf, box in zip(classes, confs, boxes.astype(np.uint16)):
            color = self.colors(class_id)
            label = self.labels_name[language][class_id]

            draw_box(result_img, label, color, conf, hide_conf, *box)

        if save_path is not None:
            cv2.imwrite(save_path, result_img)

        return result_img

    def postprocess(
        self, output: tuple[np.ndarray], ratio: float, pad: tuple[float, float]
    ) -> tuple[np.ndarray, ...]:
        """
        Convert raw output to arrays of classes, confidence and boxes.
        """
        return self.nms(output[0], ratio, pad)

    def __call__(self, img: np.ndarray, raw: bool = False) -> tuple[np.ndarray, ...]:
        img, ratio, pad = letterbox(img, self.input_size)
        raw_outputs = self.forward_pass(img)
        if raw:
            return raw_outputs

        return self.postprocess(raw_outputs, ratio, pad)


class YoloOnnxSegmentation(YoloOnnx):
    def render(
        self,
        img: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
        save_path: str | None = None,
        hide_conf: bool = True,
        language: str = "en",
    ) -> np.ndarray:
        """
        Render image with provided boxes, masks and classes.

        :param img: input image (h, w, 3)
        :param classes: the array of class_ids with shape (n, ), n is number of boxes after nms
        :param confs: the array of confs with shape (n, )
        :param boxes: the array of boxes with shape (n, 4)
        :param masks: the array on masks with shape (mask_height, mask_width, n)
        :param save_path: save/to/file.jpg
        :param hide_conf: hide confidence score from the labels
        :param language: label language
        :return: rendered image
        """
        result_img = img.copy()
        result_img = draw_masks(result_img, masks, [self.colors(i) for i in classes])
        for class_id, conf, box in zip(classes, confs, boxes.astype(np.uint16)):
            color = self.colors(class_id)
            label = self.labels_name[language][class_id]

            draw_box(result_img, label, color, conf, hide_conf, *box)

        if save_path is not None:
            cv2.imwrite(save_path, result_img)

        return result_img

    def postprocess(
        self,
        output: tuple[np.ndarray, np.ndarray],
        ratio: float,
        pad: tuple[float, float],
        shape: tuple[int, int],
        retina_masks: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Convert raw output to arrays of classes, confidence, boxes and masks.
        """
        output, protos = output
        classes, confs, boxes, masks = self.nms(output, ratio, pad, return_masks=True)
        masks = process_masks(
            protos[0], masks, boxes, self.input_size, shape, pad, retina_masks
        )

        return classes, confs, boxes, masks

    def __call__(
        self, img: np.ndarray, raw: bool = False, retina_masks: bool = False
    ) -> tuple[np.ndarray, ...]:
        shape = img.shape[:2]  # (h, w)
        img, ratio, pad = letterbox(img, self.input_size)
        raw_outputs = self.forward_pass(img)
        if raw:
            return raw_outputs

        return self.postprocess(raw_outputs, ratio, pad, shape, retina_masks)
