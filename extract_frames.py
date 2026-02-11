import cv2

def extract_frames(video_path, max_frames=30):
    vid = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frames.append(frame)
    return frames
