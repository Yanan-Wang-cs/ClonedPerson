# Pipeline Guidance
This is the guiding document for the project in "[Cloning Outfits from Real-World Images to 3D Characters for Generalizable Person Re-Identification](https://arxiv.org/pdf/2204.02611.pdf)". 


## Table of Contents

- [Pedestrian detection](#pedestrian-detection)
- [Person view qualification by pose detection](#pose-detection)
- [Clothes and keypoint detection](#clothes-detection)
- [Similarity-Diversity Expansion](#cluster)
- [Generate 3D characters](#characters)
- [Unity3D Simulation and Rendering](#rendering)
- [Image cropping.](#cropping)

## Pedestrian detection

1. Prepare environment according to [Pedestron](https://github.com/hasanirtiza/Pedestron/blob/master/INSTALL.md).
2. Download the model [epoch_19.pth.stu](https://drive.google.com/file/d/1Cw9loOUBhLJ4HYcw298V3ozfxON3ZOFN/view?usp=sharing) and put it into the folder "1Pedestron/models_pretrained"
3. Put the image to be detected into the folder of the root directory. (eg img).
4. Modify demo.sh, the last two parameters are the test folder and the result folder respectively
5. run the command: "sh demo.sh"

### Person view qualification by pose detection
1. Prepare environment according to [MMPose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md).
2. Download the model [hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth](https://drive.google.com/file/d/1TpnPTXITd9q6Dz7xCDBvdgU7d-L55ndM/view?usp=sharing) and put it into the folder "2Pose/models"
3. Put the images in folder (eg img) in the same directory as mmpose
4. Modify demo.py, update the "imgFolder", "resultFolder", and "pointsFile"
5. run the command: "python demo.py"

### Clothes and keypoint detection
1. Prepare environment according to [PIPNet](https://github.com/jhb86253817/PIPNet).
2. Download the model [epoch_12.pth](https://drive.google.com/file/d/14V2olxULzVo5b7iUAip3t8UQqjvM8E6M/view?usp=sharing) and put it into the folder "3Clothes/mmdetection/deppFashion2_multigpu"
3. Download the folder [snapshots](https://drive.google.com/drive/folders/17Qbkc0W3-0S_cMMkNvMWnWBEJ_tmJK8Y?usp=sharing) and put it into the folder "3Clothes/PIPNet_rmnb"
4. Put the images in folder (eg img) in the same directory as mmdetection
5. Modify demo.py, update the "imgFolder", "resultFolder", and "pointsFile"
6. run the command: "python demo.py"

### Similarity-Diversity Expansion

1. Prepare environment according to [QAConv](https://github.com/ShengcaiLiao/QAConv).
2. Download the model [checkpoint.pth.tar](https://drive.google.com/file/d/1YH9k_xLRCfPv5EQcyLQWBE6xuUytk0Wa/view?usp=sharing) and put it into the folder "4cluster/QAConv"
3. Put the images into the folder "4cluster/QAConv/Data/cluster/img"
4. Modify demo.sh, update the "save_data_path" and "eps".
5. run the command: "sh demo.sh"

## Generate 3D characters



