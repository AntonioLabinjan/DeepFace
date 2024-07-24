!pip install opencv-python
!pip install deepface
!pip install matplotlib
!pip install yt-dlp

# predicta dob, spol, rasu, emocije
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import json
import time
from yt_dlp import YoutubeDL

def display_image(img, title='Image'):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def analyze_frame(frame):
    results = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'])
    if not isinstance(results, list):
        results = [results]
    for result in results:
        face = result["region"]
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{result['age']} y/o, {result['gender']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"{result['dominant_race']}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"{result['dominant_emotion']}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame, results

def download_video(url, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, 'downloaded_video.mp4')
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return os.path.join(output_path, 'downloaded_video.mp4')

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_rate

def compile_video(frames, frame_rate, output_path):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def main():
    # YouTube video URL
    url = "https://youtu.be/l4Lyr_WuW5g"
    output_dir = "output_directory"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download the video
    video_path = download_video(url, output_dir)

    # Step 2: Extract frames from the video
    frames, frame_rate = extract_frames(video_path)

    # Step 3: Analyze and annotate each frame
    annotated_frames = []
    start_time = time.time()
    results = []
    for i, frame in enumerate(frames):
        annotated_frame, frame_results = analyze_frame(frame)
        annotated_frames.append(annotated_frame)
        results.append(frame_results)
        print(f"Processed frame {i+1}/{len(frames)}")

    analysis_time = time.time() - start_time

    # Step 4: Compile the processed frames back into a video
    output_video_path = os.path.join(output_dir, "annotated_video.mp4")
    compile_video(annotated_frames, frame_rate, output_video_path)

    # Step 5: Save the analysis results
    results_path = os.path.join(output_dir, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Analysis Time: {analysis_time:.2f} seconds")

if __name__ == "__main__":
    main()
