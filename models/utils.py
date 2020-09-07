import os
import torch
import numpy as np


def restore(model, resume_path: str) -> None:
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))


def compute_features(dataloader, model, N):
    batch = 32
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        print (i, '/', len(dataloader), end='\r')
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch: (i + 1) * batch] = aux
        else:
            # special treatment for final batch
            features[i * batch:] = aux

    return features
