---

# License Plate Detection and Recognition

This project implements a complete pipeline for automatic license plate detection and recognition using YOLOv8 for object detection and an OCR model for character recognition. The system can accurately detect license plates from images or video streams and extract readable license numbers, making it suitable for smart parking systems, traffic monitoring, and security surveillance.

## Overview

- Stage 1: License Plate Detection
  - Utilizes the state-of-the-art YOLOv8 object detection model.
  - Detects and localizes license plates with high accuracy and real-time performance.
  - Supports both images and live video streams.

- Stage 2: Optical Character Recognition (OCR)
  - After detecting license plates, the cropped plate images are passed to an OCR model.
  - The OCR module extracts alphanumeric characters from license plates.
  - Optionally applies image preprocessing techniques (grayscale conversion, denoising, thresholding) to improve recognition accuracy.
---

## Dataset

- Dataset from roboflow

---

## Requirements

The following libraries and frameworks are required for running the project:

* Python 3.10
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* h5py
* tqdm
* wandb
  ...

To install these dependencies, you can use the provided `requirements.txt` file.

### Example:

```
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/sonho4ng/License-Plate-Detection](https://github.com/sonho4ng/License-Plate-Detection
   cd License-Plate-Detection
   ```

2. **Install dependencies:**

   If you're using a `requirements.txt` file, run:

rewrite the description and overview so that it is suitable for this project'
