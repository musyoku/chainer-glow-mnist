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
    if hyperparams.num_image_channels != 1:
        images = np.broadcast_to(images, (images.shape[0], ) + image_size +
                                 (hyperparams.num_image_channels, ))
    images = preprocess(images, hyperparams.num_bits_x)

    dataset = glow.dataset.Dataset(images)
    iterator = glow.dataset.Iterator(dataset, batch_size=2)

    print(tabulate([["#image", len(dataset)]]))

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    total = args.num_steps + 2
    fig = plt.figure(figsize=(4 * total, 4))
    subplots = []
    for n in range(total):
        subplot = fig.add_subplot(1, total, n + 1)
        subplots.append(subplot)

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        while True:
            for data_indices in iterator:
                x = to_gpu(dataset[data_indices])
                x += xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
                factorized_z_distribution, _ = encoder.forward_step(x)

                factorized_z = []
                for (zi, mean, ln_var) in factorized_z_distribution:
                    factorized_z.append(zi)

                z = encoder.merge_factorized_z(factorized_z)
                z_start = z[0]
                z_end = z[1]

                z_batch = [z_start]
                for n in range(args.num_steps):
                    ratio = n / (args.num_steps - 1)
                    z_interp = ratio * z_end + (1.0 - ratio) * z_start
                    z_batch.append(args.temperature * z_interp)
                z_batch.append(z_end)
                z_batch = xp.stack(z_batch)

                rev_x_batch, _ = decoder.reverse_step(z_batch)
                for n in range(args.num_steps):
                    rev_x_img = make_uint8(rev_x_batch.data[n + 1], num_bins_x)
                    subplots[n + 1].imshow(rev_x_img, interpolation="none")

                x_start_img = make_uint8(x[0], num_bins_x)
                subplots[0].imshow(x_start_img, interpolation="none")

                x_end_img = make_uint8(x[-1], num_bins_x)
                subplots[-1].imshow(x_end_img, interpolation="none")

                plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--num-steps", "-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
