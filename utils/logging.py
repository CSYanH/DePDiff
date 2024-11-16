import torch
import shutil
import os
import torchvision.utils as tvu
import cv2


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    #img = img.permute(2, 1, 0)
    #cv2.imwrite(file_directory, img.detach().cpu().numpy())
    tvu.save_image(img, file_directory)
    #input_img = cv2.imread(file_directory, -1)
    #input_img = cv2.cvtColor(input_img, cv2.COLOR_HSV2BGR)
    #cv2.imwrite(file_directory, input_img)
    #print(input_img.shape)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
