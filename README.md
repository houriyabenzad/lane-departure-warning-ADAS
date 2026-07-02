# Lane Departure Warning System - ADAS

A computer vision-based Lane Departure Warning System developed with Python and OpenCV.

## Objective

The goal of this project is to detect road lane markings from images and videos and provide ADAS pipeline for lane departure warning systems.

## Features

- Road image and video processing
- Grayscale conversion
- Gaussian blur filtering
- Canny edge detection
- Region of Interest selection
- Hough Transform lane detection
- Left and right lane line estimation
- Visual lane overlay on road images/videos

## Technologies

- Python
- OpenCV
- NumPy
- Computer Vision
- ADAS

## Project Pipeline

1. Load road image or video
2. Convert frame to grayscale
3. Apply Gaussian blur
4. Detect edges using Canny
5. Apply Region of Interest mask
6. Detect lane segments using Hough Transform
7. Estimate left and right lane lines
8. Display detected lanes on the original frame

## Limitations

The current version works better on straight roads with clear lane markings. Accuracy may decrease with curved roads, shadows, poor lighting, or unclear road markings.

## Run

```bash
pip install -r requirements.txt
python main.py
