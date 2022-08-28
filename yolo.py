"""
CSCI-B657 Assignment 2, 5/4/21
Group Members:
Gavin Hemmerlein – ghemmer
Ethan Eldridge - etmeldr
Link to repo: https://github.iu.edu/cs-b657-sp2021/tdash-etmeldr-ghemmer-alavania-wilswang-a1/tree/master/finalProject
Link to Ultralytics model tutorial: https://github.com/ultralytics/yolov5/issues/36

This project will focus on an optimization of the YOLO algorithm via hyperparameter tuning. 
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

Scriterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)

# Images

imgs = ["images/IMG_4383.JPG"]
# Inference
results = model(imgs)

# Results
results.print()  
results.show()  # or .show()

# Data

print(results.pandas().xyxy[0])  # print img1 predictions (pixels)
#                   x1           y1           x2           y2   confidence        class
# tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
#         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
#         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])