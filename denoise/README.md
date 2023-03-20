
## Training
- Download the Datasets of DIV2K and LSDIR

- Generate image patches from full-resolution training images
```
python generate_patches_data.py --ps 128 --num_patches 300 --num_cores 10
```

- Generate image patches from full-resolution valid images
```
python generate_patches_data.py --ps 32 --num_patches 300 --num_cores 10
```

- Train the model with default arguments by running

```
python train.py
```
