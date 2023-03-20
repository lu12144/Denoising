

# Image Denosing

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Quick Run

To test the pre-trained models of Denoising, download pretrain_model and put it on denoise/pretrained_models and run:
```
python demo.py --input_dir path_to_images --result_dir save_images_here --task Task_Name
```


## Results of competition
please download here


## Citation
Solution is based on Multi-Stage Progressive Image Restoration of Syed Waqas Zamir et al. CVPR 2021. 
