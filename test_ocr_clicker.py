from time import sleep
import numpy as np
from modules.ocrclicker import OCRClicker
from modules.ppdocr import ocr_recognition
from pycvt import draw_bounding_boxes


import imageio.v3 as iio


bot = OCRClicker(
    window_title="开局托儿所",
    ocr_func=lambda x: ocr_recognition(x, "http://10.8.0.3:8080/ocr"),
)


if "x" in bot:
    bot.click("x")
    sleep(1)