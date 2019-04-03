[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# FaceAttribute
Six face attributes (Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young) predication from a single face image.

PyTorch implementation. Trained using celebA dataset.

## Dependencies
- Python 3.6+ (Anaconda)
- PyTorch-0.2.0 +
- scipy, numpy, sklearn etc.
- OpenCV3 (Python)

Tested on Ubuntu 14.04 LTS, Python 3.6 (Anaconda), PyTorch-0.3.0, CUDA 8.0, cuDNN 5.0

## Usage
### Data Preprocessing
detMTCNN_celebA.py

AttrListGen.py

### Data loader
dataloadercelebA.py

### Model Training
TrainAttrPreRes18V0.py

TrainAttrPreV0.py

### Models
AttrPreModelRes18_256V0.py

AttrPreModelRes34_256V0.py

### Model evaluation
AttrEvaRes18_256V0.py

AttrEvaRes34_256V0.py

### Results
Focal Loss:

Attractive 0.8231

EyeGlasses 0.9980

Male  0.9721
  
MouthOpen  0.9407
  
Smiling   0.9200

Young   0.8776
