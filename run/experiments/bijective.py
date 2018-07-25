import os
import sys
import argparse

import chainer
import chainer.functions as cf
import cupy
import numpy as np
import matplotlib.pyplot as plt

from chainer.backends import cuda
from tabulate import tabulate

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

    images = chainer.datasets.mnist.get_mnist(withlabel=False)[0]
    images = 255.0 * np.asarray(images).reshape((-1, ) + image_size + (1, ))
    images = preprocess(images, hyperparams.num_bits_x)

    dataset = glow.dataset.Dataset(images)
    iterator = glow.dataset.Iterator(dataset, batch_size=1)

    print(tabulate([["#image", len(dataset)]]))

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    fig = plt.figure(figsize=(8, 4))
    left = fig.add_subplot(1, 2, 1)
    right = fig.add_subplot(1, 2, 2)

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        while True:
            for data_indices in iterator:
                x = to_gpu(dataset[data_indices])
                x += xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
                factorized_z_distribution, _ = encoder.forward_step(x)

                factorized_z = []
                for (zi, mean, ln_var) in factorized_z_distribution:
                    factorized_z.append(zi)

                # for zi in factorized_z:
                #     noise = xp.random.normal(
                #         0, 0.2, size=zi.shape).astype("float32")
                #     zi.data += noise
                rev_x, _ = decoder.reverse_step(factorized_z)
                print(
                    xp.mean(x), xp.std(x), " ->", xp.mean(rev_x.data),
                    xp.std(rev_x.data))

                x_img = make_uint8(x[0], num_bins_x)
                rev_x_img = make_uint8(rev_x.data[0], num_bins_x)

                left.imshow(x_img, interpolation="none")
                right.imshow(rev_x_img, interpolation="none")

                plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
