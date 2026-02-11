import cv2
import numpy as np
from utils.extract_frames import extract_frames
from utils.gradcam import generate_heatmap


def localize_video(video_path, k=3):
    """
    Returns up to k frames overlaid with heatmaps.
    Each output is (frame_index, BGR_overlay_image)
    """
    frames = extract_frames(video_path, max_frames=20)  # pull up to 20 frames
    results = []

    if len(frames) == 0:
        return results

    # Pick frames spaced across the video
    idxs = np.linspace(0, len(frames) - 1, k).astype(int)

    for idx in idxs:
        frame = frames[idx]  # BGR uint8
        heat = generate_heatmap(frame)  # float32 [0..1]

        heat_u8 = (heat * 255).astype("uint8")
        heat_rgb = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heat_rgb, 0.4, 0)

        results.append((idx, overlay))

    return results
