import cv2
import numpy as np

def find_blast_area_refined(video_path, frame_skip=30, min_area=500, save_img=True, heatmap_filename = None):
    """
    Identifies ROIs using frame differencing with frame skipping to focus on larger changes.

    Args:
        video_path (str): Path to the video.
        scale_factor (int): Factor to downscale the frames.
        frame_skip (int): Number of frames to skip between analyses.
        min_area (int): Minimum area for contours to be considered as ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    previous_frame = None
    heatmap_accumulator = None
    frame_count = 0
    
    print(frame_skip)
    while True:
        for _ in range(frame_skip - 1):
            ret = cap.grab()
            if not ret:
                break

        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is not None:
            frame_diff = cv2.absdiff(previous_frame, gray_frame)
            _, fgmask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fgmask = np.zeros_like(fgmask)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    cv2.drawContours(fgmask, [cnt], -1, 255, -1)

            height, width = fgmask.shape
            distance_weights = np.linspace(1, 0.5, height).reshape(height, 1)
            weighted_fgmask = fgmask * distance_weights

            if heatmap_accumulator is None:
                heatmap_accumulator = np.zeros_like(weighted_fgmask, dtype=np.float32)
            heatmap_accumulator += weighted_fgmask

        previous_frame = gray_frame
        frame_count += 1

    normalized_heatmap = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET)

    cv2.imwrite(heatmap_filename, heatmap)

    cap.release()
    return heatmap

def extract_single_bounding_box(heatmap_path, original_video_path, buffer_area=100, min_area=500, check=True, output_frame_path="bounding_box_check.jpeg"):
    """
    Extracts a single bounding box that encompasses all detected areas from the heatmap and overlays it on one frame of the original video.

    Args:
        heatmap_path (str): Path to the heatmap image.
        original_video_path (str): Path to the original video.
        scale_factor (int): Scale factor used during heatmap generation.
        min_area (int): Minimum area for contours to be considered.
        output_frame_path (str): Path to save the output frame with bounding box.
    """
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    if heatmap is None:
        return

    _, binary_heatmap = cv2.threshold(heatmap, 40, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min, y_min, x_max, y_max = float("inf"), float("inf"), float("-inf"), float("-inf")
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    x_min = max(0, x_min-buffer_area)
    y_min = max(0, y_min-buffer_area)
    x_max = min(frame_width, x_max+buffer_area)
    y_max = min(frame_height, y_max+buffer_area)
    
    cropped_width = x_max - x_min
    cropped_height = y_max - y_min

    ret, frame = cap.read()
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(output_frame_path, frame)
    cap.release()
    
heatmap_path = "heatmap_352_103.jpeg"
original_video_path = "352_103.MP4"
find_blast_area_refined(original_video_path,heatmap_filename= heatmap_path)
output_frame_path = "352_103_box.jpeg"
extract_single_bounding_box(heatmap_path, original_video_path, buffer_area =30, output_frame_path= output_frame_path)