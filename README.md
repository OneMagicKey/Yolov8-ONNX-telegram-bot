[//]: # (![GitHub top language]&#40;https://img.shields.io/github/languages/top/OneMagicKey/Yolov8-ONNX-telegram-bot?style=flat-square&#41; ![]&#40;https://img.shields.io/github/repo-size/OneMagicKey/Yolov8-ONNX-telegram-bot?style=flat-square&#41; [![GitHub license]&#40;https://img.shields.io/github/license/OneMagicKey/Yolov8-ONNX-telegram-bot?style=flat-square&#41;]&#40;https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot/blob/master/LICENSE&#41;)

# YOLOv8-ONNX-telegram-bot
Telegram bot for object detection and instance segmentation using YOLOv5/YOLOv8 implemented on python + openCV DNN + ONNX.

- [x] ENG / RU languages
- [x] Object Detection / Instance Segmentation
- [x] Supports webhooks to deploy as webservice
- [x] Supports pooling to run bot locally

## Run bot locally
1) Create telegram bot with BotFather
2) Clone the repo: `git clone https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot.git`
3) Install all the dependencies: `pip install -r requirements.txt`
4) Add `TELEGRAM_TOKEN` provided by Botfather to the python environment
5) Run `bot.py`

## Deploy on [render](https://render.com)
1) Create telegram bot with BotFather
2) Build web service with the following options:
   * Repository - `https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot`
   * Branch - `webhooks-render`
   * Build Command - `pip install -r requirements.txt`
   * Start Command - `python bot.py`
3) Add `TELEGRAM_TOKEN` provided by Botfather to the render environment
4) Deploy the app

## Reference

1) [Ultralytics](https://github.com/ultralytics/ultralytics)
2) [yolov5_onnx_cv](https://github.com/brucefay1115/yolov5_onnx_cv)
