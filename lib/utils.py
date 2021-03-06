import os
import datetime
import json
import numpy as np
import pickle
import json
from shutil import copyfile
from torch.utils.data.sampler import Sampler
from typing import Dict, List, Tuple


def save_losses(train_losses, test_losses, path):
    with open(path, 'w') as f:
        json.dump({
            'train': train_losses,
            'test': test_losses,
        }, f, indent=4)


def visualize(
        data      : Dict,
        path      : str,
        acc_limit : float = 0.3,
) -> str:

    if not os.path.exists(path):
        os.makedirs(path)

    d = datetime.datetime.now()
    dir_name = os.path.join(path, 'exp_{}-{}-{}_{}-{}-{}'.format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    os.mkdir(dir_name)

    for key in data:
        if data[key]['max_size'] / data[key]['size'] > acc_limit:
            dir_path = os.path.join(dir_name, str(key) + '_' + data[key]['max_cls'])
        else:
            dir_path = os.path.join(dir_name, str(key))
        os.mkdir(dir_path)
        for i, path in enumerate(data[key]['paths']):
            cls, name = path.rsplit('/', 2)[1:]
            dest = os.path.join(dir_path, str(i) + '_' + cls + '_' + name)
            copyfile(path, dest)
    return dir_name


def visualize_prediction(
        predictions,
        path,
):
    import datetime
    import os
    from shutil import copyfile
    d = datetime.datetime.now()
    dir_name = os.path.join(path, 'exp_{}-{}-{}_{}-{}-{}'.format(d.year, d.month, d.day, d.hour, d.minute, d.second))
    os.mkdir(dir_name)

    for key in predictions:
        dir_path = os.path.join(dir_name, predictions[key]['cls_str'][0])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dest = os.path.join(dir_path, key.rsplit('/', 2)[1] + '_' + key.rsplit('/', 1)[1])
        copyfile(key, dest)


def accuracy(data: Dict, acc_limit: float = 0.3) -> None:
    correct = 0
    wrong = 0
    for key in data:
        if data[key]['max_size'] / data[key]['size'] < acc_limit: continue

        correct += data[key]['max_size']
        wrong += data[key]['size'] - data[key]['max_size']

    print ('Accuracy: ', correct / (correct + wrong))


def check_classes(save_dir: str, data: List[Tuple[str, int]]) -> None:
    paths = {x.split('_', 1)[1] for x in os.listdir(save_dir) if '_' in x}
    all_paths = {x.rsplit('.', 1)[0].rsplit('/', 2)[1] for x, _ in data}
    print (all_paths.difference(paths))


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def get_available_classes(data: Dict, thr_perc: float = 0.3) -> Dict[int, str]:

    avaliable_cls = {}

    for key in data.keys():
        if data[key]['max_size'] / data[key]['size'] > thr_perc:
            avaliable_cls[key] = data[key]['max_cls']

    return avaliable_cls


def clean_predictions(I: np.ndarray, available_classes: Dict[int, str]) -> np.ndarray:
    available_classes_indexes = set(available_classes.keys())
    new_I = []
    for i in I:
        new_i = []
        for x in i:
            if len(new_i) >= 5: break
            if str(x) in available_classes_indexes:
                new_i.append(x)
        if len(new_i) < 5:
            new_i.extend([-1] * (5 - len(new_i)))
        new_I.append(new_i)
    return np.array(new_I)


def save_predictions_imgs(predictions: Dict[str, Dict], save_dir: str) -> None :
    for x in predictions:
        if len(predictions[x]['cls_str']) == 0:
            continue
        dir_path = os.path.join(save_dir, predictions[x]['cls_str'][0])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        copyfile(x, os.path.join(dir_path, x.rsplit('/', 2)[1] + '_' + x.rsplit('/', 1)[1]))


def save_predictions_json(predictions: Dict[str, Dict], save_dir: str, file_name: str = 'predictions.json') -> None :
    for x in predictions:
        predictions[x]['cls_id'] = predictions[x]['cls_id'].tolist()
    with open(os.path.join(save_dir, file_name), 'w')as f:
        json.dump(predictions, f)
