import argparse
import json
import torch
import numpy as np
from math import ceil
from PIL import Image
from torchvision import models
from torch import nn
from collections import OrderedDict

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, default=5, help='Choose top K matches as int.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping from categories to real names.')
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image_path):
    image = Image.open(image_path)
    # Resize and crop the image
    size = 256
    crop_size = 224
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_height = int(size / aspect_ratio)
        image = image.resize((size, new_height))
    else:
        new_width = int(size * aspect_ratio)
        image = image.resize((new_width, size))
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(image) / 255.0
    np_image = (np_image - mean) / std
    # Transpose the color channel
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def predict(image_path, label_path, model, device, topk=5):
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
    with torch.no_grad():
        model.eval()
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
    probabilities = torch.exp(output)
    top_p, top_class_idx = probabilities.topk(topk, dim=1)
    top_probs = top_p.squeeze().cpu().numpy()
    top_classes = [list(model.class_to_idx.keys())[i] for i in top_class_idx.squeeze().cpu().numpy()]
    with open(label_path, 'r') as f:
        cat_to_name = json.load(f)
    top_flower_names = [cat_to_name[class_idx] for class_idx in top_classes]
    return top_probs, top_classes, top_flower_names

def print_results(probs, flowers):
    for i, (prob, flower) in enumerate(zip(probs, flowers), 1):
        print(f"Rank {i}: Flower: {flower}, Likelihood: {prob*100:.2f}%")

def main():
    args = arg_parser()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    top_probs, top_classes, top_flower_names = predict(args.image, args.category_names, model, device, args.top_k)
    print_results(top_probs, top_flower_names)

if __name__ == '__main__':
    main()

