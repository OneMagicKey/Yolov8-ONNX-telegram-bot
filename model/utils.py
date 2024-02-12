import cv2
import numpy as np


def xywh2box(boxes: np.ndarray, scale: float, padw: float, padh: float) -> np.ndarray:
    """
    Convert (x,y,w,h) to (x,y,w,h) in original image size.
    """
    boxes[..., [0, 1]] -= 0.5 * boxes[..., [2, 3]]
    boxes[..., [0, 1]] -= np.array([padw, padh])
    boxes /= scale

    return boxes


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert (x,y,w,h) to (x1,y1,x2,y2).
    """
    boxes[..., [2, 3]] += boxes[..., [0, 1]]

    return boxes


def xyxy2new_shape(boxes: np.ndarray, old_shape, new_shape) -> np.ndarray:
    """
    Convert boxes coordinates (x1,y1,x2,y2) to the new shape.
    """
    h_old, w_old = old_shape
    h_new, w_new = new_shape

    boxes[..., [0, 2]] *= w_new / w_old
    boxes[..., [1, 3]] *= h_new / h_old

    return boxes


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # Copy from https://github.com/ultralytics/yolov5/blob/master/utils/segment/general.py
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    :param masks: should be a size (h, w, n) array of masks
    :param boxes: should be a size (n, 4) array of bbox coords in relative point form
    :return: cropped masks (h, w, n)
    """
    h, w, n = masks.shape
    x1, y1, x2, y2 = np.split(boxes.transpose()[None, ...], 4, 1)  # x1 shape(1, 1, n)  # fmt: skip

    r = np.arange(w, dtype=x1.dtype)[None, :, None]  # rows shape(1, w, 1)
    c = np.arange(h, dtype=x1.dtype)[:, None, None]  # cols shape(h, 1, 1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def draw_box(
    img: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    confidence: float,
    hide_conf: bool,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    """
    Draw bounding box and label on the input image.
    """
    label = f"{label}" if hide_conf else f"{label} {int(confidence * 100)}%"
    th, fs = 2, 0.65  # thickness, fontScale
    lt, ff = cv2.LINE_AA, cv2.FONT_HERSHEY_COMPLEX  # lineType, fontFace
    w, h = cv2.getTextSize(label, ff, fs, th)[0]
    x1, x2, y1, y2 = map(np.int32, (x1, x2, y1, y2))

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th, lt)

    # Draw label
    outside = y1 - th - h < 0
    text_up = y1 + h if outside else y1 - th  # top border of the text
    text_bg = (y1 + th + h if outside else y1 - th - h)  # top border of the text background  # fmt: skip

    cv2.rectangle(img, (x1 - th, y1 - th), (x1 - th + w, text_bg), color, -1, lt)
    cv2.putText(img, label, (x1 - th, text_up), ff, fs, (255, 255, 255), th, lt)


def draw_masks(
    img: np.ndarray,
    masks: np.ndarray,
    colors: list[tuple[int, int, int]],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw masks on top of the input image.

    :param img: image with shape (h, w, 3)
    :param masks: masks with shape (mask_height, mask_width, n), n is number of masks after nms
    :param colors: list of BGR color tuples, (n, 3)
    :param alpha: masks weight in the result image, 0 <= alpha <= 1
    :return: input image with masks drawn on top
    """
    # No masks after nms
    if not masks.shape[-1]:
        return img

    h, w = img.shape[:2]
    alpha = np.float32(alpha)
    colors = np.asarray(colors, dtype=np.uint8)[None, None, ...]  # (1, 1, n, 3)
    masks = masks[..., None]  # (mask_height, mask_width, n, 1)

    # apply argmax over the mask scores at each pixel
    indices = masks.argmax(axis=2, keepdims=True)
    colored_mask = np.take_along_axis((masks > 0.5) * colors, indices, axis=2)  # (mask_height, mask_width, 1, 3)  # fmt: skip
    colored_mask = colored_mask.squeeze() * alpha  # (mask_height, mask_width, 3)
    colored_mask = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_AREA)  # (h, w, 3)  # fmt: skip

    overall_mask = colored_mask.any(axis=-1, keepdims=True)  # union of the masks (h, w, 1)  # fmt: skip
    img = img * (1 - overall_mask * alpha) + colored_mask

    return img


def letterbox(
    img: np.ndarray, input_size: tuple
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize the image to predefined model input size preserving the aspect ratio, pad if needed.

    :param img: image of shape (h, w, 3)
    :param input_size: predefined size of the model input, (model_height, model_width)
    :return: resized image of shape (model_height, model_width, 3), scale and pad params
    """
    height, width, _ = img.shape

    scale = min(input_size[0] / height, input_size[1] / width)

    new_width, new_height = map(round, [width * scale, height * scale])
    padh, padw = (input_size[0] - new_height) / 2, (input_size[1] - new_width) / 2

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    top, bottom = map(round, [padh - 0.1, padh + 0.1])
    left, right = map(round, [padw - 0.1, padw + 0.1])
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return img, scale, (padw, padh)


def process_masks(
    protos: np.ndarray,
    mask_coefs: np.ndarray,
    boxes: np.ndarray,
    input_size: tuple[int, int],
    original_img_shape: tuple[int, int],
    pad: tuple[float, float],
    retina_masks: bool = False,
) -> np.ndarray:
    # Inspired by https://github.com/ultralytics/yolov5/blob/master/utils/segment/general.py
    """
    Combine mask_coefficients and protos to create masks.

    :param protos: protos, (mask_dim, mask_height, mask_width)
    :param mask_coefs: (n, mask_dim), n is number of masks after nms
    :param boxes: boxes with shape (n, 4)
    :param input_size: predefined size of the model input, (model_height, model_width)
    :param original_img_shape: image spatial shape before preprocessing, (h, w)
    :param pad: initial pad for restore masks to original size
    :param retina_masks: process masks in high resolution, default is False to save memory
    :return: masks with shape (model_height - 2*padh, model_width-2*padw, n)
    """
    # No masks after nms
    if not mask_coefs.shape[0]:
        return np.zeros((*original_img_shape, 0), dtype=bool)

    c, mh, mw = protos.shape

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    masks = sigmoid(np.einsum("nc,chw->nhw", mask_coefs, protos))  # (n, mh, mw)

    gain = min(mh / input_size[0], mw / input_size[1])  # 0.25 predefined by yolo
    padw, padh = pad[0] * gain, pad[1] * gain

    # remove padding from the masks
    top, left = round(padh - 0.01), round(padw - 0.01)
    bottom, right = mh - round(padh + 0.01), mw - round(padw + 0.01)
    masks = masks[:, top:bottom, left:right].transpose(1, 2, 0)

    if retina_masks:
        # better quality, but higher memory and time consumption.
        up_h, up_w = original_img_shape
    else:
        # upsample masks to shape (model_height - 2*padh, model_width - 2*padw)
        up_w, up_h = map(round, [(right - left) / gain, (bottom - top) / gain])
        boxes = xyxy2new_shape(boxes.copy(), original_img_shape, (up_h, up_w))

    masks = cv2.resize(masks, dsize=(up_w, up_h), interpolation=cv2.INTER_LINEAR)  # (up_h, up_w, n)  # fmt: skip
    masks = masks[..., None] if len(masks.shape) == 2 else masks
    masks = crop_mask(masks, boxes)  # (up_h, up_w, n)

    return masks
