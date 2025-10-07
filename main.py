import cv2
import os
import numpy as np
import sys

# --- Configuration ---
IMAGE_PATH = 'src-img\\t1.jpg'  
OUTPUT_DIR = 'out-img'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# DNN Model Files
PROTOTXT_PATH = 'deploy.prototxt.txt'
CAFFEMODEL_PATH = 'res10_300x300_ssd_iter_140000.caffemodel'
CONFIDENCE_THRESHOLD = 0.5

# Quality Filters
MIN_CROP_DIMENSION = 80

# Blur detection function
# Blur Detection Constant (10 to 300)
BLUR_THRESHOLD = 150.


def is_blurry(image, threshold=BLUR_THRESHOLD):
    if image is None or image.size == 0:
        return True
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

# Detection Setup
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
except:
    print("Could not load DNN model files.")
    print("Please ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' are downloaded.")
    sys.exit()

def recheck_face_validity(cropped_face_img, dnn_net, confidence_threshold=CONFIDENCE_THRESHOLD):
    if cropped_face_img is None or cropped_face_img.size == 0:
        return False, 0.0
    
    (h, w) = cropped_face_img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(cropped_face_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    max_confidence = 0.0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
    
        if confidence > max_confidence:
            max_confidence = confidence
    # returns true when max confidence if more than the set threshold
    return max_confidence >= confidence_threshold, max_confidence


# Loading the cascade classifier    
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print(f"Error: Cascade classifier not loaded. Check path: {CASCADE_PATH}")
    exit()

# Reading Image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: Could not read image file {IMAGE_PATH}")
    exit()

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray_image, 
    scaleFactor=1.05, 
    minNeighbors=5,   
    minSize=(50, 50)  # Increased minSize slightly to reduce very small false positives
)

total_detected = len(faces)
auto_rejected_count = 0
auto_rejected_dnn=0
auto_rejected_small=0
saved_count = 0
print(f"Detected {total_detected} potential faces. Starting interactive review...")

#Save delete loop
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face region from the *original color image*
    cropped_face = image[y:y + h, x:x + w]
    
    # 1.Automatic Resolution check
    if w < MIN_CROP_DIMENSION or h < MIN_CROP_DIMENSION:
        auto_rejected_count += 1
        print(f"[AUTO REJECTED] Face {i+1} too small (WxJ: {w}x{h}). Skipping...")
        continue
    
    # 2. Blur check
    is_blurry_flag, variance = is_blurry(cropped_face, BLUR_THRESHOLD)
    if is_blurry_flag:
        auto_rejected_count += 1
        print(f"[AUTO REJECTED] Face {i+1} is blurry (Variance: {variance: 2f}). Skipping...")
        continue

    # 3. DNN Recheck
    is_valid_face, dnn_confidence = recheck_face_validity(cropped_face, net, CONFIDENCE_THRESHOLD)
    if not is_valid_face:
        auto_rejected_dnn += 1
        print(f"[AUTO REJECTED] Face {i+1} failed DNN recheck (Max Confidence: {dnn_confidence*100:.2f}%). Skipping...")
        continue

    # 4. Maual Review
    # Create a unique window for the cropped face
    window_name = f"Face {i+1}/{total_detected} - [S]ave or [D]iscard"
    cv2.imshow(window_name, cropped_face)
    
    # Wait for a key press  's' or 'd'
    
    while True:
        key = cv2.waitKey(0) & 0xFF # Wait for a key press
        
        if key == ord('s'):
            # Save the cropped image
            face_filename = os.path.join(OUTPUT_DIR, f'face_{saved_count:04d}_cropped.jpg')
            cv2.imwrite(face_filename, cropped_face)
            saved_count += 1
            print(f"-> Saved face {i+1} as {face_filename}")
            break
            
        elif key == ord('d'):
            # Delete the image
            print(f"-> Discarded face {i+1}")
            break

        elif key == ord('q'):
            # Quit program
            print("Quitting process.")
            cv2.destroyAllWindows()
            exit()
            
        else:
            print("Invalid key. Press 's' to Save, 'd' to Discard, or 'q' to Quit.")
            continue
            
    cv2.destroyWindow(window_name)

cv2.destroyAllWindows()
print("-" * 30)
print(f"Review complete.")
print(f"Total detected: {total_detected}.")
print(f"Auto-rejected (too small): {auto_rejected_small}.")
print(f"Auto-rejected (blurry): {auto_rejected_count}.")
print(f"Auto-rejected (DNN fail): {auto_rejected_dnn}.")
print(f"Manually Reviewed: {total_detected-auto_rejected_count-auto_rejected_dnn-auto_rejected_small}.")
print(f"Total saved: {saved_count}")   


