import numpy as np
import torch
from torch import nn
from torch import optim


class Experiment:
	def __init__(self, model, train_loader, valid_loader, model_name="checkpoint.pt"):
		"""
		initialize an experiment
		:param model: model, eg -- torchvision.models.resnet18(pretrained=True)
		:param train_loader: dataloader of training dataset
		:param valid_loader: dataloader of validation dataset
		:param model_name: where to save the model
		"""
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = model
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.model_name = model_name
		# criterion and optimizer
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

	def load_model(self):
		self.model.load_state_dict(torch.load(self.model_name))

	def train(self, num_epoch):
		self.load_model()
		for epoch in range(num_epoch):
			train_loss, valid_loss = 0, 0
			for i, (image, label) in enumerate(self.train_loader):
				image, label = image.to(self.device), image.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.model(image)
				loss = self.criterion(outputs, label)
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()

	def validate(self):
		for i, (image, label) in enumerate(self.valid_loader):
			image, label = image.to(self.device), label.to(self.device)