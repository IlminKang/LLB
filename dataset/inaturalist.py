import json
import numpy as np
import torch.utils.data as data
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets.folder import default_loader

ANN_FILE = {
    "train": "train2018.json",
    "val": "val2018.json",
    "test": "test2018.json",
}


def load_taxonomy(ann_data, classes):
    # loads the taxonomy data and converts to ints
    tax_levels = [
        "id",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
    ]
    taxonomy = {}

    if "categories" in ann_data.keys():
        num_classes = len(ann_data["categories"])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data["categories"]]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0] * len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class iNaturalist2018(data.Dataset):
    def __init__(self, root_dir, transform, split="train", full_info=False):
        root = root_dir
        mode = split
        """ A Dataset for iNaturalist data.

        Args:
            data ([type]): Parent class.
            root (str or Path): Path to the root folder.
            mode (str, optional): Defaults to "train". Establishing if the
                dataset is of type `train`, `validation` or `test` and loads
                the coresponding data.
            transform (torchvision.transforms.Transform, optional): Defaults
                to None. A transform function fore preprocessing and
                augmenting images.
            full_info (bool, optional): Defaults to False. If `True` the
                loader will return also the `taxonomic_class` and the `img_id`.
        """

        self._mode = mode
        self._full_info = full_info

        # make pathlib.Paths
        ann_file = ANN_FILE[mode]
        try:
            self._root = root
            self.annotations_path = root / ann_file
        except TypeError:
            self._root = root = Path(root)
            self._ann_file = ann_file = root / ann_file

        # load annotations
        print(f"\t|-iNaturalist: loading annotations from: {ann_file}.")
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self._img_paths = [root / aa["file_name"] for aa in ann_data["images"]]

        # if we dont have class labels set them to '0'
        if "annotations" in ann_data.keys():
            self._classes = [a["category_id"] for a in ann_data["annotations"]]
        else:
            self._classes = [0] * len(self._img_paths)

        self._num_classes = len(set(self._classes))

        if full_info:
            # get image id
            self._img_ids = [aa["id"] for aa in ann_data["images"]]
            # load taxonomy
            self._taxonomy, self._classes_taxonomic = load_taxonomy(
                ann_data, self._classes
            )

        # image loading, preprocessing and augmentations
        self.loader = default_loader

        self.transform = transform

        # print out some stats
        print(f"\t\t|-iNaturalist: found {len(self._img_paths)} images.")
        print(f"\t\t|-iNaturalist: found {len(set(self._classes))} classes.")

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        img = self.loader(self._img_paths[index])
        species_id = self._classes[index]  # class

        if self.transform:
            img = self.transform(img)

        if self._full_info:
            # we can also return some additionl info
            img_id = self._img_ids[index]
            tax_id = self._classes_taxonomic[species_id]

            return img, species_id, tax_id, img_id
        return img, species_id

    def __str__(self):
        details = f"len={len(self)}, mode={self._mode}, root={self._root}"
        return f"iNaturalistDataset({details})"

    def __len__(self):
        return len(self._img_paths)