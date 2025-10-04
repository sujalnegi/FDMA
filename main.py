# Creation DATE: 4 October 2025
# 1st Hour
import cv2
print("This is a test Face detection application")
print("Testing Methods")    

# Reading image
image = cv2.imread("src-img\\1.jpeg")

if image is None:
    print("Error: Image not found")
else:
    print("Image loaded Successfully!")
    #Display the read image
    cv2.imshow("FDMA", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Image read will be showed 

#Converting image to grayscale for faster processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayScale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#get image resolution and colorscale info
h, w, c = image.shape
print(f"Image Loaded:\nHeight: {h}\nWidth: {w}\nChannels: {c}")


#Store output cropped face
cv2.imwrite("face1.jpg", image)