from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import Literal

import cv2
import numpy as np
import onnxruntime

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
    """
    Base class for YOLO models

    Attributes:
        input_size (tuple[int, int]): the input size of the model in (height, width).
        conf (float): confidence threshold for non-maximum suppression.
        iou (float): intersection over union threshold for non-maximum suppression.
        version (Literal[5, 8]): version of YOLO model (5 or 8).
        model (onnxruntime.InferenceSession): ONNX inference session for the model.
        colors (Colors): instance of Colors class for coloring boxes and masks.
        labels_name (defaultdict): mapping of class labels in different languages.
    """

    __slots__ = (
        "input_size",
        "conf",
        "iou",
        "version",
        "model",
        "colors",
        "labels_name",
    )

    def __init__(
        self,
        checkpoint: str,
        input_size: tuple[int, int] = (640, 640),
        conf: float = 0.25,
        iou: float = 0.45,
        version: Literal[5, 8] = 8,
    ) -> None:
        self.input_size = input_size  # (h, w)
        self.conf = conf
        self.iou = iou
        self.version = version
        self.model = self.build_model(checkpoint)

        self.colors = Colors()
        self.labels_name = defaultdict(
            lambda: COCO_names_en, en=COCO_names_en, ru=COCO_names_ru
        )

    def build_model(self, checkpoint: str) -> onnxruntime.InferenceSession:
        """
        Create a model from the ONNX checkpoint.
        """
        model = onnxruntime.InferenceSession(
            checkpoint, providers=["CPUExecutionProvider"]
        )

        return model

    def forward_pass(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray]:
        """
        Create a blob from the image and pass it through the model.

        :param img: np.array of shape (h, w, 3)
        :return: (boxes_array) or (boxes_and_masks_array, protos_array)
        """
        (h, w) = self.input_size
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (w, h), swapRB=True, crop=False)  # n, c, h, w  # fmt: skip

        output_names = [layer.name for layer in self.model.get_outputs()]
        input_name = self.model.get_inputs()[0].name
        output = self.model.run(output_names, {input_name: blob})

        return output

    def print_results(
        self,
        classes: np.ndarray,
        confs: np.ndarray,
        boxes: np.ndarray,
        masks: None | np.ndarray = None,
        language: Literal["en", "ru"] = "en",
    ):
        """
        Create a list of tuples containing information about detected objects.

        :return: list of tuples BoxInfo("class_id", "class_name", "conf", and "box")
        """
        BoxInfo = namedtuple("BoxInfo", ["class_id", "class_name", "conf", "box"])
        result_list = [
            BoxInfo(class_id, self.labels_name[language][class_id], conf, box)
            for class_id, conf, box in zip(classes, confs, boxes)
        ]

        return result_list

    def version_handler(self, output: np.ndarray, nc: int) -> np.ndarray:
        """
        Convert output of the model to a single format.

        :param output: an array with boxes from the model's output
        :param nc: number of classes
        :return: output with shape (num_objects, 4+num_classes+num_masks)
        """
        match self.version:
            case 8:
                output = output.transpose((0, 2, 1)).squeeze(0)
            case 5:
                output = output[output[..., 4] > self.conf]
                output[..., 5 : 5 + nc] *= output[..., 4:5]  # conf = obj_conf * cls_conf  # fmt: skip
                output = np.delete(output, 4, axis=1)  # remove obj_conf to preserve the format   # fmt: skip
            case _:
                output = output.squeeze(0)

        return output

    def get_colors(
        self, classes: np.ndarray, color_scheme: Literal["equal", "random"] = "equal"
    ) -> list[tuple[int, int, int]]:
        """
        Create a list of colors to color bounding boxes and masks.

        :param classes: the array of class labels with shape (n, )
        :param color_scheme: the 'equal' or 'random' color for objects of the same class
        :return: list of colors with length n
        """
        if color_scheme == "random":
            colors_ids = np.random.randint(self.colors.n, size=len(classes))
        else:
            colors_ids = classes

        colors = [self.colors(i) for i in colors_ids]

        return colors

    def nms(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        return_masks: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Post-processes the output from a YOLOv5/YOLOv8 model to extract classes,
        confidences, bounding boxes, and optionally, mask coefficients.

        This function processes the raw output from the model, filters detections using
        non-maximum suppression, and adjusts the bounding box coordinates based on the
        provided ratio and padding values.

        :return: classes, confidences, boxes and (optional) mask coefficients
        """
        nc = len(self.labels_name["en"])
        output = self.version_handler(output, nc)

        boxes, confs, masks_coefs = np.split(output, [4, 4 + nc], axis=1)
        classes = confs.argmax(axis=-1)
        confs = confs[np.arange(classes.shape[-1]), classes]
        boxes = xywh2box(boxes, ratio, padw=pad[0], padh=pad[1])

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.iou)
        indices = list(indices)
        classes, confs, boxes = classes[indices], confs[indices], boxes[indices]

        if return_masks:
            return classes, confs, xywh2xyxy(boxes), masks_coefs[indices]

        return classes, confs, xywh2xyxy(boxes)

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
    __slots__ = ()

    def render(
        self,
        img: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        boxes: np.ndarray,
        save_path: str | None = None,
        hide_conf: bool = False,
        color_scheme: Literal["equal", "random"] = "equal",
        language: Literal["en", "ru"] = "en",
    ) -> np.ndarray:
        """
        Render image with provided boxes and classes.

        :param img: input image (h, w, 3)
        :param classes: the array of class_ids with shape (n, ), n is number of boxes after nms
        :param confs: the array of confs with shape (n, )
        :param boxes: the array of boxes with shape (n, 4)
        :param save_path: save/to/file.jpg
        :param hide_conf: hide confidence score from the labels
        :param color_scheme: 'equal' or 'random' color for objects of the same class
        :param language: label language
        :return: rendered image
        """
        result_img = img.copy()
        colors = self.get_colors(classes, color_scheme)

        for color, class_id, conf, box in zip(colors, classes, confs, boxes):
            label = self.labels_name[language][class_id]
            draw_box(result_img, label, color, conf, hide_conf, *box)

        if save_path is not None:
            cv2.imwrite(save_path, result_img)

        return result_img

    def postprocess(
        self, output: tuple[np.ndarray], ratio: float, pad: tuple[float, float]
    ) -> tuple[np.ndarray, ...]:
        """
        Convert raw output to arrays of classes, confidences and boxes.
        """
        return self.nms(output[0], ratio, pad)

    def __call__(
        self, img: np.ndarray, raw: bool = False, **kwargs
    ) -> tuple[np.ndarray, ...]:
        img, ratio, pad = letterbox(img, self.input_size)
        raw_outputs = self.forward_pass(img)
        if raw:
            return raw_outputs

        return self.postprocess(raw_outputs, ratio, pad)


class YoloOnnxSegmentation(YoloOnnx):
    __slots__ = ()

    def render(
        self,
        img: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
        save_path: str | None = None,
        hide_conf: bool = False,
        color_scheme: Literal["equal", "random"] = "equal",
        language: Literal["en", "ru"] = "en",
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
        :param color_scheme: 'equal' or 'random' color for objects of the same class
        :param language: label language
        :return: rendered image
        """
        result_img = img.copy()
        colors = self.get_colors(classes, color_scheme)

        result_img = draw_masks(result_img, masks, colors)

        for color, class_id, conf, box in zip(colors, classes, confs, boxes):
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
        Convert raw output to arrays of classes, confidences, boxes, and masks.
        """
        output, protos = output
        classes, confs, boxes, mask_coefs = self.nms(
            output, ratio, pad, return_masks=True
        )
        masks = process_masks(
            protos[0], mask_coefs, boxes, self.input_size, shape, pad, retina_masks
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
