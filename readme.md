Unofficial implementation of paper "PanoPoint: Self-Supervised Feature Points Detection and Description for 360° Panorama" published in 2023 CVPR workshop.

# Usage
1. `python train_first.py` to train the model on synthetic panoramic images in the first stage.
2. `python train.py` to train the model on real-world panoramic images.
3. `python detect.py` for test
4. We provide model weights trained on [CVPG dataset](https://github.com/semihorhan/semseg-outdoor-pano): `panopoint_final.pth`