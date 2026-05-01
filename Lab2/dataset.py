import os
import tifffile as tiff
from torch.utils.data import Dataset

class RoadsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])

        mask_name = os.path.splitext(self.images[index])[0] + ".tif"
        mask_path = os.path.join(self.mask_dir, mask_name)


        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)
        mask = (mask > 0).astype("float32")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask