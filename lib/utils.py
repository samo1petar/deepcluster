import os
import datetime
from shutil import copyfile
from typing import Dict, List, Tuple


def visualize(
        data      : Dict,
        path      : str   = '/home/david/Datasets/Flickr',
        acc_limit : float = 0.3,
) -> str: # TODO fix path
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
            cls, name = path.rsplit('.', 1)[0].rsplit('/', 2)[1:]
            dest = os.path.join(dir_path, str(i) + '_' + cls + '_' + name + '.png')
            copyfile(path, dest)
    return dir_name


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
