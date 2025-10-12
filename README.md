# Face Extractor Web Application

![Face Extractor Screenshot](https://i.ibb.co/5h5mzTqj/image.png)

Face Extractor is a powerful and intuitive web application built with Flask and OpenCV that allows users to automatically detect, filter, and extract high-quality human faces from images, videos, and live webcam streams.

The application is designed with a multi-stage filtering pipeline to ensure that the extracted faces are clear, well-framed, and genuine, making it an ideal tool for collecting data for computer vision projects.

## Features

* **Multiple Input Sources:**
    * **Image Upload:** Process static images (`.jpg`, `.png`, `.jpeg`).
    * **Video Upload:** Extract faces from video files (`.mp4`, `.mov`, `.avi`). The application processes every 5th frame to ensure efficiency.
    * **Live Webcam Capture:** Record a 10-second video clip directly from your webcam for real-time face extraction.
* **Advanced Filtering Pipeline:**
    * **Resolution Check:** Discards detected faces that are smaller than a specified dimension (80x80 pixels) to ensure quality.
    * **Blur Check:** Uses Laplacian Variance to automatically reject blurry or out-of-focus faces.
    * **DNN Model Re-check:** Employs a high-accuracy Deep Neural Network (DNN) model to eliminate false positives.
    * **Duplicate Removal:** (For Video/Webcam) Utilizes image hashing to intelligently identify and discard duplicate faces, ensuring a unique dataset.
* **User-Friendly Interface:**
    * A clean, modern, and fully responsive UI that works on all devices.
    * Live image preview and file name display before submission.
    * Easy download of all extracted faces as a single `.zip` file.

## Technologies Used

* **Backend:** Python, Flask
* **Computer Vision:** OpenCV, NumPy
* **Deep Learning Model:** Caffe-based Face Detection Model
* **Duplicate Detection:** ImageHash, Pillow
* **Frontend:** HTML5, CSS3 (Flexbox), JavaScript

## Local Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

* Python 3.7+
* `pip` (Python package installer)

### 2. Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/sujalnegi/FDMA.git
cd FDMA
```
### 3. Create a Virtual Environment

It is highly recommended to create a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 4. Install Dependencies

Install all the required Python libraries using `pip`.

```bash
pip install Flask opencv-python numpy werkzeug imagehash Pillow
```
Alternatively, you can use requirements.txt `pip install -r requirements.txt`

### 5. Run the Application

Once the setup is complete, you can start the Flask development server:
```bash
python app.py
```
Now, open your web browser and navigate to the following address:

```
http://127.0.0.1:5000/
```
You should see the Face Extractor application running!

### How to Use
1. Navigate to the desired page using the navbar: "Image", "Video", or "WebCam".
2. Choose a file or start recording from your webcam.
3. Select the quality filters you wish to apply. The "Remove Duplicate Faces" option is available for video and webcam modes.
4. Click the "Extract Faces Now" or "Record 10 Seconds" button.
5. After processing, you will be redirected to the results page where you can preview all extracted faces.
6. Click the "Download All Faces (.zip)" button to save the results to your computer.



