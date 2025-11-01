import argparse
import os
from ultralytics import YOLO
import cv2
import numpy as np
from itertools import combinations
from math import hypot
from datetime import datetime

MODEL_PATH = "yolov8_weights/yolov8n.pt"  
SAVE_DIR = "runs"
NEAR_MISS_PIX_THRESHOLD = 80  
DETECT_CLASSES = None  
MIN_BOX_AREA = 100 