import os
import glob
from PIL import Image
from torch.utils import data
from torchvision import transforms


class Market1501(data.Dataset):
	"""
	Market-1501 dataset:
	"""
	def __init__(self, root, mode="train", transform=None):
		super(Market1501, self).__init__()
		self.root = os.path.abspath(root)
		self.mode = mode
		self.transform = transform
		folder_name = "bounding_box_" + mode
		self.folder = os.path.join(self.root, folder_name)
		self.images = glob.glob(os.path.join(self.folder, "*.jpg"))

	def __getitem__(self, ind):
		img_path = self.images[ind]
		img_name = img_path.split("/")[-1]
		img = Image.open(img_path).convert("RGB")
		label = int(img_name[:4])
		# use default transform
		if self.transform is None:
			self.transform = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor()
			])
		img = self.transform(img)
		return img, label

	def __len__(self):
		return len(self.images)