import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--input_dir', default='temp/valid_data', type=str, help='Input images')
parser.add_argument('--result_dir', default='temp/MPRNet-main/output', type=str, help='Directory for results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


task    = args.task
inp_dir = args.input_dir
out_dir = args.result_dir
# task = 'Denoising'
# inp_dir = '/home/wyx/temp/input/temp_data'
# out_dir = '/home/wyx/temp/output/output_temp'
# tile = 360
# tile_overlap = 32

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                + glob(os.path.join(inp_dir, '*.JPG'))
                + glob(os.path.join(inp_dir, '*.png'))
                + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding model architecture and weights
weights = os.path.join(task, "pretrained_models", "model_"+task.lower()+".pth")
# weights = '/home/wyx/temp/MPRNet-main/Denoising/pretrained_models/model_denoising.pth'

load_file = run_path(os.path.join(task,'MPRNet.py'))
# load_file = run_path('/home/wyx/temp/MPRNet-main/Denoising/MPRNet.py')
model = load_file['MPRNet']()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['state_dict'])
model.eval()



img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")

with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
    

        img = Image.open(file_).convert('RGB')
        
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()

        # img_1 = load_img(file_)

        # input_1 = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)


        # print("print img:\n")
        # print(img)
        # print(input_)

        # print("print img_1:\n")
        # print(img_1)
        # print(input_1)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if args.tile is None:
        # if tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            # tile = min(tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch[0])

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch[0])
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:height,:width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        # stx()
        save_img((os.path.join(out_dir, f+'.png')), restored)

    print(f"\nRestored images are saved at {out_dir}")
