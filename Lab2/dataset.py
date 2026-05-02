import os
from pathlib import Path

import tifffile as tiff
from torch.utils.data import Dataset


def make_tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


class RoadsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, tile_size=512, stride=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride
        self.images = sorted(
            name for name in os.listdir(self.image_dir)
            if name.endswith((".tif", ".tiff"))
        )
        self.tiles = self._make_tiles()

    def _image_shape(self, image_name):
        with tiff.TiffFile(self.image_dir / image_name) as tif:
            shape = tif.pages[0].shape
        return shape[0], shape[1]

    def _make_tiles(self):
        tiles = []
        for image_name in self.images:
            height, width = self._image_shape(image_name)
            y_starts = make_tile_starts(height, self.tile_size, self.stride)
            x_starts = make_tile_starts(width, self.tile_size, self.stride)
            tiles.extend((image_name, y, x) for y in y_starts for x in x_starts)
        return tiles

    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, index):
        image_name, y, x = self.tiles[index]
        image_path = self.image_dir / image_name

        mask_name = os.path.splitext(image_name)[0] + ".tif"
        mask_path = self.mask_dir / mask_name


        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)
        image = image[y:y + self.tile_size, x:x + self.tile_size]
        mask = mask[y:y + self.tile_size, x:x + self.tile_size]
        mask = (mask > 0).astype("float32")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask
