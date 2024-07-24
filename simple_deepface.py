!pip install opencv-python deepface matplotlib

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import json
import time

def display_image(img, title='Image'):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

        
def analyze_face(img_path, output_dir):
    start_time = time.time()
    
    img = cv2.imread(img_path)
    results = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'])

    analysis_time = time.time() - start_time

    if not isinstance(results, list):
        results = [results]
    
    for result in results:
        face = result["region"]
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{result['age']} y/o, {result['gender']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, f"{result['dominant_race']}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img, f"{result['dominant_emotion']}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    display_image(img, title='Analyzed Image')    
    return img, results, analysis_time

def save_results(img, results, output_dir, filename):
    output_img_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_img_path, img)

    results_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def main():
    # Hardcoded image path
    img_path = "human.jpg"
    output_dir = "output_directory"
    os.makedirs(output_dir, exist_ok=True)

    img, results, analysis_time = analyze_face(img_path, output_dir)
    save_results(img, results, output_dir, os.path.basename(img_path))

    for i, result in enumerate(results):
        print(f"Face {i+1}:")
        print(f"  Age: {result['age']}")
        print(f"  Gender: {result['gender']}")
        print(f"  Dominant Race: {result['dominant_race']}")
        print(f"  Dominant Emotion: {result['dominant_emotion']}")
        print()
    
    print(f"Analysis Time: {analysis_time:.2f} seconds")

if __name__ == "__main__":
    main()
