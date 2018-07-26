import os
import sys
import argparse

import chainer
import chainer.functions as cf
import cupy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from chainer.backends import cuda
from tabulate import tabulate
from sklearn.manifold import TSNE

sys.path.append(os.path.join("..", ".."))
import glow

sys.path.append("..")
from model import Glow, to_cpu, to_gpu
from hyperparams import Hyperparameters


def make_uint8(array, bins):
    if array.ndim == 4:
        array = array[0]
    if (array.shape[2] == 3):
        return np.uint8(
            np.clip(
                np.floor((to_cpu(array) + 0.5) * bins) * (255 / bins), 0, 255))
    return np.uint8(
        np.clip(
            np.floor((to_cpu(array.transpose(1, 2, 0)) + 0.5) * bins) *
            (255 / bins), 0, 255))


def preprocess(image, num_bits_x):
    num_bins_x = 2**num_bits_x
    if num_bits_x < 8:
        image = np.floor(image / (2**(8 - num_bits_x)))
    image = image / num_bins_x - 0.5
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    elif image.ndim == 4:
        image = image.transpose((0, 3, 1, 2))
    else:
        raise NotImplementedError
    return image


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x
    image_size = (28, 28)

    _, test = chainer.datasets.mnist.get_mnist()
    images = []
    labels = []
    for entity in test:
        image, label = entity
        images.append(image)
        labels.append(label)
    labels = np.asarray(labels)
    images = 255.0 * np.asarray(images).reshape((-1, ) + image_size + (1, ))
    if hyperparams.num_image_channels != 1:
        images = np.broadcast_to(images, (images.shape[0], ) + image_size +
                                 (hyperparams.num_image_channels, ))
    images = preprocess(images, hyperparams.num_bits_x)

    # images = images[:200]
    # labels = labels[:200]

    sections = len(images) // 100
    dataset_image = np.split(images, sections)

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    fig = plt.figure(figsize=(8, 8))
    t_sne_inputs = []

    with chainer.no_backprop_mode():
        for n, image_batch in enumerate(dataset_image):
            x = to_gpu(image_batch)
            x += xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
            factorized_z_distribution, _ = encoder.forward_step(x)

            factorized_z = []
            for (zi, mean, ln_var) in factorized_z_distribution:
                factorized_z.append(zi)

            z = encoder.merge_factorized_z(
                factorized_z, factor=hyperparams.squeeze_factor)
            z = z.reshape((-1, hyperparams.num_image_channels * 28 * 28))
            z = to_cpu(z)
            t_sne_inputs.append(z)

    t_sne_inputs = np.asanyarray(t_sne_inputs).reshape(
        (-1, hyperparams.num_image_channels * 28 * 28))
    print(t_sne_inputs.shape)

    z_reduced = TSNE(
        n_components=2, random_state=0).fit_transform(t_sne_inputs)
    print(z_reduced.shape)

    plt.scatter(
        z_reduced[:, 0], z_reduced[:, 1], c=labels, s=5, cmap="Spectral")
    plt.colorbar()
    plt.savefig("scatter.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
