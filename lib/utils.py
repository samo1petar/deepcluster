import os
import datetime
from shutil import copyfile


def visualize(data, path='/home/david/Datasets/Flickr', acc_limit: float = 0.3): # TODO fix path
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


def accuracy(data, acc_limit: float = 0.3):
    correct = 0
    wrong = 0
    for key in data:
        if data[key]['max_size'] / data[key]['size'] < acc_limit: continue

        correct += data[key]['max_size']
        wrong += data[key]['size'] - data[key]['max_size']

    print ('Accuracy: ', correct / (correct + wrong))