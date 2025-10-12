# Face Extractor Web Application

![Face Extractor Screenshot](https://ibb.co/3Y9PcwwW)

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
git clone [https://github.com/sujalnegi/FDMA.git](https://github.com/sujalnegi/FDMA.git)
cd FDMA
