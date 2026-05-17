import torch

with open('model_keys.txt', 'w') as f:
    img = torch.load('../models/model_epoch_1.pth', map_location='cpu')
    f.write("=== IMAGE MODEL KEYS ===\n")
    for key in img.keys():
        f.write(key + "\n")

    vid = torch.load('../models/model_epoch_3.pth', map_location='cpu')
    f.write("\n=== VIDEO MODEL KEYS ===\n")
    for key in vid.keys():
        f.write(key + "\n")

print("Done — open model_keys.txt to see all keys")