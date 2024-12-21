import cv2
import numpy as np

path = "352_108.MP4"
cap = cv2.VideoCapture(path)

print("Frame Rate:", int(cap.get(cv2.CAP_PROP_FPS)))
print("Total Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print(f"Video Resolution: {frame_width}x{frame_height}")

cap.release()