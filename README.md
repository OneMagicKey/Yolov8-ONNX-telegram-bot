<h2 align="center">YOLOv8-ONNX-telegram-bot</h2>

<p align="center">
<a href="https://t.me/yolov8_onnx_bot"><img src="https://img.shields.io/badge/Telegram-yolov8__onnx__bot-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<img src ="https://img.shields.io/github/repo-size/OneMagicKey/Yolov8-ONNX-telegram-bot">
<a href="https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/actions/workflows/ci.yml"><img src="https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<h2 align="center"><img width="768" src="example/example.png"></h2>

A Telegram bot for object detection and instance segmentation using YOLOv5/YOLOv8/YOLOv10/YOLOv11,
implemented in Python + OpenCV + ONNXRuntime.

Additionally, the bot can be deployed on the [Render](https://render.com) cloud platform
for free.

The **webhook version** of the bot requires around 380 MB of RAM and includes 2 **quantized** models:
`yolov11s` and `yolov11s-seg`. The **polling version** includes 7
**quantized** models: `yolov5s`, `yolov8s`, `yolov10s`, `yolov11s`, `yolov5s-seg`, `yolov8s-seg`, and `yolov11s-seg`,
and requires approximately 700 MB of RAM.
## Features

- [x] English / Russian language support
- [x] Object Detection / Instance Segmentation
- [x] Quantized YOLOv5, YOLOv8, YOLOv10, and YOLOv11 models
- [x] Support webhooks to deploy as a webservice on the Render platform ([webhooks-render](https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/blob/webhooks-render/) branch)
- [x] Support polling to run bot locally ([master](https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/blob/master/) branch)

## List of Bot Commands

* `/start` - start the bot
* `/help` - list of available commands
* `/language` - set language preferences
* `/model` - select a model
* `/color_scheme` - select a color scheme (`'equal'` or `'random'` colors for detected objects of the same class)
* `/retina_masks` - enable high-quality segmentation masks

These commands are automatically added to the bot at startup.

## Run the Bot on Your Local Machine (Polling Version)
### Run via Python Environment (Requires Python >= 3.10)

1) Create a telegram bot with [BotFather](https://telegram.me/BotFather)
2) Clone the repo and install all the dependencies:

   ```bash
   git clone https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot.git
   cd Yolov8-ONNX-telegram-bot
   pip install -r requirements.txt
   ```

3) Add the `TELEGRAM_TOKEN` provided by BotFather to the Python environment or replace
`TOKEN` in `src/bot.py` with your token
4) Run `python src/bot.py`

### Run via Docker
1) Create a telegram bot with [BotFather](https://telegram.me/BotFather)
2) Clone the repo:

   ```bash
   git clone https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot.git
   cd Yolov8-ONNX-telegram-bot
   ```

3) Replace a dummy `TELEGRAM_TOKEN` in the Dockerfile with your telegram token
4) Build the image and run it:
   ```bash
   docker build -t yolo-bot .
   docker run yolo-bot
   ```

## Deploy on the [Render](https://render.com) Cloud Platform (Webhooks Version)
### Automatic Deployment
1) Create a telegram bot with [BotFather](https://telegram.me/BotFather)
2) Create a Render account
3) Click on the button below to deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/tree/webhooks-render)

### Manual Deployment
1) Create a telegram bot with [BotFather](https://telegram.me/BotFather)
2) Create a Render account
3) Go to `New` -> `Web Service` -> `Build and deploy from a Git repository`
4) Build a webservice with the following options:
   * Public Git repository - `https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot`
   * Branch - `webhooks-render`
   * Runtime - `Python 3`
   * Build Command - `pip install -r requirements.txt`
   * Start Command - `python src/bot.py`
   * Instance Type = `Free`
5) Add Environment Variables:
   * Name - `TELEGRAM_TOKEN`, value `{YOUR_TELEGRAM_TOKEN}`
   * Name - `PYTHON_VERSION`, value `3.10.13`
6) Deploy the app

> Note: Instances under the Render free plan will spin down after about 15 minutes of inactivity.
To keep the bot running, configure a [cron-job](https://cron-job.org/) to send a POST request to the bot every 10 minutes.

## Adding New Models

To add `n/s/m/l/x` versions of Yolo to the bot or to adjust the model's image input size,
follow these steps:

1) Export a model to ONNX format:

   ```bash
   pip install ultralytics
   ```

   ```python
   from ultralytics import YOLO

   # Load a model
   model_name = 'yolov8s'  # [yolov8s, yolov8m, yolov8s-seg, yolov8m-seg, ...]
   model = YOLO(f"{model_name}.pt")

   # Image size
   height, width = (640, 640)  # [(640, 480), (480, 640), ...]

   # Export the model
   model.export(format='onnx', opset=14, simplify=True, optimize=True, imgsz=(height, width))
   ```

2) Place the resulting `.onnx` file in either the `src/checkpoints/detection` or `src/checkpoints/segmentation`
   folder, depending on the model type
3) Add the model to the `model_list` in `src/bot.py` using the format:

   `ModelInfo(type, name, input_size, conf, iou, version)`

   ```python
   # Example:

   ModelInfo("detection", "yolov8s", (640, 640), version=8),  # small detection model
   ModelInfo("segmentation", "yolov8m-seg", (640, 480), version=8),  # medium segmentation model with rectangular input size
   ```

## References

1) [Ultralytics](https://github.com/ultralytics/ultralytics)
2) [yolov5_onnx_cv](https://github.com/brucefay1115/yolov5_onnx_cv)
