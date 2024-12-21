import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

def detect_explosion_start(video_path, min_ssim=61, var_ssim=3, output_folder="output"):
    """
    Detects the start of an explosion in a video by analyzing frame differences.

    Parameters:
    - video_path (str): Path to the video file.
    - min_ssim (float): Minimum SSIM difference threshold.
    - var_ssim (float): Variance of SSIM difference to ensure localized change.
    - output_folder (str): Folder to save output frames and heatmaps.

    Returns:
    - int: The frame number where the explosion starts, or -1 if not detected.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to load video.")
        return -1
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count > 450:
            

            ssim_score, ssim_diff = ssim(prev_gray, gray, full=True)
            ssim_diff = (ssim_diff * 255).astype("uint8")


            min_ssim_diff = np.min(ssim_diff)
            max_ssim_diff = np.max(ssim_diff)
            mean_ssim_diff = np.mean(ssim_diff)
            var_ssim_diff = np.var(ssim_diff)
            print(frame_count, min_ssim_diff, max_ssim_diff, var_ssim_diff, mean_ssim_diff)


            if (min_ssim_diff == min_ssim) and (var_ssim_diff < var_ssim):
                print(f"Explosion detected at frame {frame_count}")

                before_frame_path = os.path.join(output_folder, f"frame_{frame_count - 1}_new_before.jpg")
                cv2.imwrite(before_frame_path, prev_frame)


                after_frame_path = os.path.join(output_folder, f"frame_{frame_count}_new_after.jpg")
                cv2.imwrite(after_frame_path, frame)

                heatmap = cv2.applyColorMap(ssim_diff, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
                heatmap_overlay_path = os.path.join(output_folder, f"frame_{frame_count}_new_heatmap_overlay.jpg")
                cv2.imwrite(heatmap_overlay_path, overlay)

                cap.release()
                return frame_count

        print(frame_count)
        prev_gray = gray
        prev_frame = frame
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("No explosion detected.")
    return -1

video_path = "328_109.MP4"
explosion_frame = detect_explosion_start(video_path, min_ssim=0, var_ssim=140, output_folder="output")
print(f"Explosion starts at frame: {explosion_frame}")