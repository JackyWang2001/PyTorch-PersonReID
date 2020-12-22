import torch.nn as nn
import torchvision


def init_MobileNetV2(num_classes):
	model = torchvision.models.mobilenet_v2(pretrained=True)
	# replace the classifier
	model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
	return model