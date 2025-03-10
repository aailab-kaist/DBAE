from torchvision.datasets import LFWPeople
import os
import numpy as np


class LFWAttribute(LFWPeople):
    def __init__(
        self,
        root: str,
        split: str = "10fold",
        image_set: str = "funneled",
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super(LFWPeople, self).__init__(root, split, image_set, "people", transform, target_transform, download)
        self.load_attributes()
        self.class_to_idx = self._get_classes()
        self.data, self.targets, self.attributes = self._get_people()

    def load_attributes(self):
        with open(os.path.join(self.root, 'lfw_attributes.txt')) as f:
            lines = f.readlines()
        self.attribute_names = lines[1].strip().split('\t')[3:]
        self.num_attributes = len(self.attribute_names)
        self.attribute_dict = {}
        for line in lines[2:]:
            processed = line.strip().split('\t')
            name = processed[0] + '-' + processed[1]
            val = [float(attr) for attr in processed[2:]]
            self.attribute_dict[name] = val

    def _get_people(self):
        data, targets, attributes = [], [], []
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()
            n_folds, s = (int(lines[0]), 1) if self.split == "10fold" else (1, 0)

            for fold in range(n_folds):
                n_lines = int(lines[s])
                people = [line.strip().split("\t") for line in lines[s + 1 : s + n_lines + 1]]
                s += n_lines + 1
                for i, (identity, num_imgs) in enumerate(people):
                    for num in range(1, int(num_imgs) + 1):
                        name = identity + '-' + str(num)
                        name = " ".join(name.split('_'))
                        if name in self.attribute_dict.keys():
                            img = self._get_path(identity, num)
                            data.append(img)
                            targets.append(self.class_to_idx[identity])
                            attributes.append(self.attribute_dict[name])
                        # else:
                        #     print(name)       # missing in attribute file hence not loaded

        return data, targets, attributes
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        attribute = self.attributes[index]
        return img, target, np.array(attribute)