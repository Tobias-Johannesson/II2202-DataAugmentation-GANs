import random

import numpy as np
import pandas as pd

import torch
import torchvision
import matplotlib.pyplot as plt

def plot_samples(data):
    """
    
    """

    classes = {0: ('akiec', 'Actinic keratoses'),
           1:('bcc' , ' basal cell carcinoma'),
           2:('bkl', 'benign keratosis-like lesions'),
           3: ('df', 'dermatofibroma'),
           4: ('nv', ' melanocytic nevi'),
           5: ('vasc', ' pyogenic granulomas and hemorrhage'),
           6: ('mel', 'melanoma'),
           }
    
    CLASSES = [classes[idx][0] for idx in range(len(classes))] # abbreviated form of classes
    CLASSES_FULL = [classes[idx][1] for idx in range(len(classes))] # Full name of classes

    sample_images = []
    N = len(CLASSES) # number of samples per class
    for class_ in classes.keys():
        image_idxs = data.label==class_
        N_ = len(data[image_idxs])
        chosen = random.sample(list(np.arange(N_)), k= N) # creating random 7 samples per class
        images = np.asarray(data[image_idxs].iloc[chosen,:-1])# grabing those random 7 samples

        for img in images:
            sample_images.append(torch.tensor(img.reshape(28, 28, 3)).permute(2, 0, 1)) # obtaining one image at a time

    s = torch.stack(sample_images) # stack all images, convert to torch.tensor for grid
    grid = torchvision.utils.make_grid(s, nrow=N) # create grid with same rows and cols

    plt.figure(figsize=(8, 8), dpi=(128)) # plot the grid
    plt.imshow(grid.permute(1,2,0))
    plt.xticks(np.linspace(14,grid.shape[2]-14, 7), labels=[f'sample {idx}' for idx in range(N)])
    plt.yticks(np.linspace(14,grid.shape[1]-14, 7), labels=[f'[{idx}] {cls_}' for idx, cls_ in enumerate(CLASSES_FULL)])
    plt.title('Sample of skin lesions in HAM10000')
    plt.legend(CLASSES_FULL)
    plt.savefig('ham10000_samples.png') # Save image as png
    plt.show(block='off')