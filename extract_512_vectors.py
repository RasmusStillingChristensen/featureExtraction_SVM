from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from torch.autograd import Variable
import os
import numpy as np
from PIL import Image
import csv
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to the trained VGG16 model')
parser.add_argument('--dataset', type=str, help='path to the dataset we want to label')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# VGG-16 Takes 224x224 images as input
transform=transforms.Compose([
                              transforms.Pad((0,0)),
                              transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Modify the classifier part of VGG16
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] 
features = list(vgg16.classifier.children())[:-1] 
features.extend([nn.Linear(num_features, 512)])
features.extend([nn.Linear(512, 60)]) # Adjust input size for the last layer
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# move model to gpu
if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs.")
    vgg16 = nn.DataParallel(vgg16)
else:
    print("Running on CPU.")
vgg16.to(device)
# upload pretrained weights from beauty labeled dataset
vgg16.load_state_dict(torch.load(opt.model))

# get output from last layer of neurons instead.
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])

vgg16.eval()

# create beauty rates lists for each image in dataset
files = []
neurons = []
images_dir = "{0}/img".format(opt.dataset)
number_of_images = len(os.listdir(images_dir))

for i, file in enumerate(sorted(os.listdir(images_dir))):

    # open image, transform and upload to gpu
    img = Image.open(os.path.join(images_dir,file)).convert('RGB')
    img = transform(img)
    img = torch.from_numpy(np.asarray(img))
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img.cuda())
    else:
        with torch.no_grad():
            img = Variable(img)
    img = torch.unsqueeze(img,0)

    # infer image to receive beauty rates
    output = vgg16(img)

    # convert output tensor into list with rounded values
    output_list = (output.data.cpu().numpy().tolist())[0]
    output_list = [round(x,10) for x in output_list] #change number of digits to optimize data size

    files.append(file)
    neurons.append(output_list)

    if (i % 100 == 0):
        print('{0}/{1} images done'.format(i,number_of_images))

# convert lists to csv lines
csv_lines = []

		
		
for j in range(0, number_of_images):
    scores = ''
    for i in range(0, 512):  # Iterate over all 512 elements
        scores += '{0},'.format(str(neurons[j][i] * 1.0))
    csv_lines.append('{0},{1}'.format(files[j], scores[:-1]))  # Remove the last comma


# write csv lines to file
csv_path = "{0}/Vectors_512_removed_layer_cropped.csv".format(opt.dataset)
with open(csv_path, "w") as csv_file:
    for line in csv_lines:
        csv_file.write(line)
        csv_file.write('\n')
