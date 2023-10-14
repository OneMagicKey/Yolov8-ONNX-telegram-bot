import cv2
import numpy as np


def xywh2box(boxes: np.ndarray, scale: float, padw: float, padh: float) -> np.ndarray:
    """ Convert (x,y,w,h) to (x,y,w,h) in original image size """
    boxes[..., [0, 1]] -= 0.5 * boxes[..., [2, 3]]
    boxes[..., [0, 1]] -= np.array([padw, padh])
    boxes /= scale

    return boxes.astype(np.uint16)


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """ Convert (x,y,w,h) to (x1,y1,x2,y2) """
    boxes[..., [2, 3]] += boxes[..., [0, 1]]

    return boxes


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # Copy from https://github.com/ultralytics/yolov5/blob/master/utils/segment/general.py
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    :param masks: should be a size [n, h, w] tensor of masks
    :param boxes: should be a size [n, 4] tensor of bbox coords in relative point form
    :return: cropped masks (n, h, w)
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n, 1, 1)

    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1, 1, w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1, h, 1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def draw_box(img: np.ndarray, label: str, color: tuple[int, int, int],
             confidence: float, hide_conf: bool, x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Draw bounding box and label on the input image.
    """
    label = f'{label}' if hide_conf else f'{label} {confidence:.2f}'
    th, fs = 2, 0.65  # thickness, fontScale
    lt, ff = cv2.LINE_AA, cv2.FONT_HERSHEY_COMPLEX  # lineType, fontFace
    w, h = cv2.getTextSize(label, ff, fs, th)[0]

    cv2.rectangle(img, (x1 - th, y1 - th), (x1 + w - th, y1 - h - th), color, -1, lt)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th, lt)
    cv2.putText(img, label, (x1 - th, y1 - th), ff, fs, (255, 255, 255), th, lt)


def draw_masks(img: np.ndarray, masks: np.ndarray, colors: list[tuple[int, int, int]],
               alpha: float = 0.4) -> np.ndarray:
    """
    Draw masks on the input image.

    :param img: image with shape (h, w, 3)
    :param masks: masks with shape (n, h, w), n is number of masks after nms
    :param colors: list of RGB color tuples, (n, 3)
    :param alpha: masks weight, 0 < alpha < 1
    :return: input image with the masks
    """
    if not masks.shape[0]:
        return img

    h, w = img.shape[:2]
    # save memory
    masks = masks.astype(np.uint8).transpose(1,2,0)
    masks = cv2.resize(masks, (640, 640), interpolation=cv2.INTER_NEAREST)
    masks = masks if len(masks.shape) > 2 else masks[..., None]

    colors = np.array(colors, dtype=np.uint8)[None, None, ...]  # (1, 1, n, 3)
    masks = masks[..., None]  # (h, w, n, 1)
    colored_masks = colors * masks  # (h, w, n, 3)
    colored_masks = colored_masks.max(axis=2)  # take only 1 color, merge channels in case of overlapping

    colored_masks = cv2.resize(colored_masks, (w, h), interpolation=cv2.INTER_NEAREST)
    img, alpha, colored_masks = map(np.float16, [img, alpha, colored_masks])
    img = np.where(colored_masks > 0, img * (1-alpha), img) + colored_masks * alpha

    return img


def letterbox(img: np.ndarray, input_size: tuple) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize the image to predefined model input size preserving the aspect ratio, pad if needed.

    :param img: image of shape (h, w, 3)
    :param input_size: predefined size of the model input, (model_height, model_width)
    :return: resized image of shape (model_height, model_width, 3)
    """
    height, width, _ = img.shape

    scale = min(input_size[0] / height, input_size[1] / width)

    new_width, new_height = map(round, [width * scale, height * scale])
    padh, padw = (input_size[0] - new_height) / 2, (input_size[1] - new_width) / 2

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    top, bottom = map(round, [padh - 0.1, padh + 0.1])
    left, right = map(round, [padw - 0.1, padw + 0.1])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))

    return img, scale, (padw, padh)


def process_masks(protos: np.ndarray, masks_in: np.ndarray, boxes: np.ndarray,
                  input_size: tuple[int, int], original_img_shape: tuple[int, int],
                  pad: tuple[float, float]) -> np.ndarray:
    # Inspired by https://github.com/ultralytics/yolov5/blob/master/utils/segment/general.py
    """
    Create image masks from output masks and protos.

    :param protos: protos, (mask_dim, mask_height, mask_width)
    :param masks_in: (n, mask_dim), n is number of masks after nms
    :param boxes: boxes with shape (n, 4)
    :param input_size: predefined size of the model input, (model_height, model_width)
    :param original_img_shape: image spatial shape before preprocessing, (h, w)
    :param pad: initial pad for restore masks to original size
    :return: masks with shape (n, h, w)
    """
    if not masks_in.shape[0]:
        # no detection
        return masks_in
    c, mh, mw = protos.shape

    def sigmoid(x: np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-x))
    masks = sigmoid(masks_in @ protos.view().reshape(c, -1)).view().reshape(-1, mh, mw)  # (n, mh, mw)

    gain = min(mh / input_size[0], mw / input_size[1])    # 0.25 predefined by yolo
    padw, padh = pad[0] * gain, pad[1] * gain

    top, left = round(padh - 0.01), round(padw - 0.01)
    bottom, right = mh - round(padh + 0.01), mw - round(padw + 0.01)
    masks = masks[:, top:bottom, left:right].transpose(1,2,0)

    masks = cv2.resize(masks, dsize=original_img_shape[::-1], interpolation=cv2.INTER_LINEAR)  # (h, w, n)
    masks = masks.astype(np.float16)
    masks = masks if len(masks.shape) > 2 else masks[..., None]
    masks = crop_mask(masks.transpose(2,0,1), boxes)  # (n, h, w)

    return masks > 0.5
