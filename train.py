import argparse
import torch
from torchvision import models, transforms, datasets
from torch import nn, optim
from collections import OrderedDict
from PIL import Image

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory.', required=True)
    parser.add_argument('--arch', type=str, help='Architecture (VGG or Densenet)', default='vgg16')
    parser.add_argument('--save_dir', type=str, help='Directory to save the checkpoint', default='checkpoint.pth')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the classifier', default=512)
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=10)
    parser.add_argument('--gpu', action="store_true", help='Use GPU + Cuda for calculations')
    args = parser.parse_args()
    return args

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader

def build_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
    else:
        print("Invalid architecture choice.")
        return None
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader)*100:.2f}%")

def save_checkpoint(model, train_data, save_dir, arch, hidden_units):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)

def main():
    args = arg_parser()
    trainloader, validloader, testloader = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device=device)
    
    save_checkpoint(model, trainloader.dataset, args.save_dir, args.arch, args.hidden_units)
    print(f"Model trained and saved to {args.save_dir}")

if __name__ == '__main__':
    main()
