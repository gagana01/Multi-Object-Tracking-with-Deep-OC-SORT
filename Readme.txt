-->code folder contains main python files.
	-->MASKRCNN.py -> Mask-R-CNN model
	-->KEYPOINT.py-> Keypoints model
	-->DeepOCSort.py-> DeepOCSort model

-->input videos folder contains video files used in this project.

-->To run the Program,


-->Method:
Open a terminal in CAP-5415-Project folder and run command, python code/MASKRCNN.py or python code/KEYPOINT.py , python code/DeepOCSort.py

For the metrics , run the file python code/metrics.py

Dependencies to install;

!pip install git+https://github.com/facebookresearch/detectron2.git@main

!pip install deep_sort_realtime

!pip install motmetrics opencv-python pandas

!sudo apt-get install tesseract-ocr
!pip install pytesseract

dependencies:
- bzip2=1.0.8=h93a5062_5
- ca-certificates=2023.7.22=hf0a4a13_0
- libexpat=2.5.0=hb7217d7_1
- libffi=3.4.2=h3422bc3_5
- libsqlite=3.44.0=h091b4b1_0
- libzlib=1.2.13=h53f4e23_5
- ncurses=6.4=h463b476_2
- openssl=3.1.4=h0d3ecfb_0
- pip=23.3.1=pyhd8ed1ab_0
- python=3.11.0=h3ba56d0_1_cpython
- readline=8.2=h92ec313_1
- setuptools=68.2.2=pyhd8ed1ab_0
- tk=8.6.13=h5083fa2_1
- tzdata=2023c=h71feb2d_0
- wheel=0.41.3=pyhd8ed1ab_0
- xz=5.2.6=h57fd34a_0