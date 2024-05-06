# INFORM-Project
Implementation of Food Image Detection and Segmentation methods for edge devices.

## Setting Up Your Environment
To run the scripts, you need a properly configured Python environment. Follow these steps to set up:

1. **Create and Activate a Virtual Environment:**

   For Conda:
   ```bash
   conda create -n myenv python=3.8
   conda activate myenv
   ```

   For venv:
   ```bash
   # Create a new virtual environment
   python -m venv myenv
   
   # Activate the environment
   # On macOS/Linux:
   source myenv/bin/activate
   # On Windows:
   myenv\Scripts\activate
   ```

2. **Install Dependencies:**
Ensure you have Python installed, along with the following packages:
- OpenCV (`opencv-python`)
- NumPy
- Scikit-Learn
- Scikit-Image
- PIL (Pillow)
- Matplotlib
- PyTorch
- YOLO (Ultralytics)
- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM.git)
- [MobileSAM](https://github.com/yourusername/MobileSAM.git)

You can install all required packages using the following command:
```bash
pip install -r requirements.txt
```

## Installation
Ensure your environment is active, and you are in the project directory.
Clone the repository to your local machine:
```
git clone https://github.com/inesruizblach/INFORM-Project.git
cd INFORM-Project
```


## Usage
This project includes two main functions for image processing:

1. **YOLO EdgeSAM Predictor**
   - This function integrates the YOLOv8 detection model with EdgeSAM or MobileSAM segmentation models.

2. **YOLO Detect and Segment**
   - This function uses the YOLOv8 model for both detection and segmentation.

### YOLO EdgeSAM Predictor
To run the YOLO EdgeSAM Predictor, use the following command:
```bash
python yolov8-sam.py <img_dir> <ann_dir> <categories_txt_file_path> <sam_model> <yolo_model_path> [--batch_size <int>] [--plot_results]
```
Example:
```bash
python yolov8-sam.py images/ annotations/ categories.txt edgesam models/yolov8_weights.pth --batch_size 200 --plot_results
```

### YOLO Detect and Segment
To run the YOLO Detect and Segment function, use the following command:
```bash
python yolov8-segment.py <img_dir> <ann_dir> <categories_txt_file_path> <yolov8_model_path> [--batch_size <int>] [--plot]
```
Example:
```bash
python yolov8-segment.py images/ annotations/ categories.txt models/yolov8_weights.pth --batch_size 200 --plot
```
Replace the paths with your actual data and model paths. Use the `--plot` option to enable plotting if the script supports it.

## Deactivating the Environment
After you finish running your scripts, deactivate the virtual environment:

```bash
deactivate
```
