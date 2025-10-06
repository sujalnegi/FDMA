import cv2
import os

# --- Configuration ---
IMAGE_PATH = 'src-img\\t1.jpg'  
OUTPUT_DIR = 'out-img'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

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
saved_count = 0
print(f"Detected {total_detected} potential faces. Starting interactive review...")

#Save delete loop
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face region from the *original color image*
    cropped_face = image[y:y + h, x:x + w]
    
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
print(f"Review complete. Total detected: {total_detected}. Total saved: {saved_count}")
