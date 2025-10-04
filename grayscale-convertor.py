import cv2
# 1. Take image path from user
print("This is the grayscale convertor")
print("-------------------------------")
path = input("Image Path: ")
image = cv2.imread(path)
if image is None:
    print("Either the image is corrupted or Specified path is wrong")
else:
    print("Processing...")

#2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#Choice
choice = input("Enter\n1. to save the image\n2. to view the image\n3. for both\nChoice: ")

if choice is '1':
    output_path = input("Enter Output Path: ")
    cv2.imwrite(output_path, gray)

elif choice is '2':
    cv2.imshow("Output Image",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif choice is '3':
    output_path = input("Enter Output Path: ")
    cv2.imwrite(output_path, gray)
    cv2.imshow("Output Image",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else: 
    print("Wrong Choice")

