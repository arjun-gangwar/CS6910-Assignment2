from torchvision.io import read_image
from torch.utils.data import Dataset

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        label = self.labels[idx]
        image = image / 255
        if image.shape[0] == 1:       # handle gray scale images
            image = image.repeat(3,1,1)
        if self.transform:
            image = self.transform(image)
        return image, label