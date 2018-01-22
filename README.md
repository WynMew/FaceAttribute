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

Attractive 0.8236

EyeGlasses 0.9972

Male  0.9704
  
MouthOpen  0.9394
  
Smiling   0.9176

Young   0.8731
