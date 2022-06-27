# VIMS-IAARC-DATATHON-2022

Source code for VIMS-IAARC-DATATHON-2022

```
python3 -m virtualenv yolact-datathon
```

```
source yolact-datathon/bin/activate
```

```
pip3 install cython
!pip3 install opencv-python pillow pycocotools matplotlib
!pip3 install torchvision==0.5.0
!pip3 install torch==1.4.0
```

```
cd yolact-datathon
git clone https://github.com/dbolya/yolact.git
rm -rf yolact/data/config.py
mv ./config.py yolact/data/
cd yolact
mkdir weights
cd weights
wget https://paingthet.com/leaves_detection_1041_100000.pth
```

Get test images

```
cd ../..

mkdir test_images

cd ..
mkdir output_images


python3 ./yolact/eval.py --trained_model=./yolact/weights/leaves_detection_1041_100000.pth --config=yolact_darknet53_leaves_custom_config --score_threshold=0.15 --top_k=15 --images=test_images:output_images
```
