<h2 align="center">YOLOv8-ONNX-telegram-bot</h2>

<p align="center">
<img src="https://img.shields.io/github/languages/top/OneMagicKey/Yolov8-ONNX-telegram-bot">
<img src ="https://img.shields.io/github/repo-size/OneMagicKey/Yolov8-ONNX-telegram-bot">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
Telegram bot for object detection and instance segmentation using YOLOv5/YOLOv8 implemented on python + openCV DNN + ONNX.

- [x] ENG / RU languages
- [x] Object Detection / Instance Segmentation
- [x] Support webhooks to deploy as webservice
- [x] Support pooling to run bot locally

## Run bot locally
1) Create telegram bot with BotFather
2) Clone the repo: `git clone https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot.git`
3) Install all the dependencies: `pip install -r requirements.txt`
4) Add `TELEGRAM_TOKEN` provided by BotFather to the python environment
5) Run `bot.py`

## Deploy on [render](https://render.com)
1) Create telegram bot with BotFather
2) Build web service with the following options:
   * Repository - `https://github.com/OneMagicKey/Yolov8-ONNX-telegram-bot`
   * Branch - `webhooks-render`
   * Build Command - `pip install -r requirements.txt`
   * Start Command - `python bot.py`
3) Add `TELEGRAM_TOKEN` provided by BotFather to the render environment
4) Deploy the app

## Example
<img width="768" src="images/example.png">

## Reference

1) [Ultralytics](https://github.com/ultralytics/ultralytics)
2) [yolov5_onnx_cv](https://github.com/brucefay1115/yolov5_onnx_cv)
