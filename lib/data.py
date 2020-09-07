import operator
import cv2
import numpy as np
import torch


def transform_to_dict(data):
    data_dict = {}

    for path, cls in data:
        label = path.rsplit('/', 2)[1]
        if cls not in data_dict:
            data_dict[cls] = {
                'paths': [path],
                'labels': {label: 1},
            }
        else:
            data_dict[cls]['paths'].append(path)
            if label not in data_dict[cls]['labels']:
                data_dict[cls]['labels'][label] = 1
            else:
                data_dict[cls]['labels'][label] += 1

    for key in data_dict:
        max_cls, max_size = max(data_dict[key]['labels'].items(), key=operator.itemgetter(1))
        data_dict[key]['max_cls'] = max_cls
        data_dict[key]['max_size'] = max_size
        data_dict[key]['size'] = sum([x for x in data_dict[key]['labels'].values()])

    return data_dict


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.stack((image[..., 0], image[..., 1], image[..., 2]), axis=0)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image_tensor = torch.from_numpy(image)
    return image_tensor
