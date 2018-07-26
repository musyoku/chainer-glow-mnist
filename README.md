[Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039)

# Usage

```
cd run
python3 train.py -levels 2 -depth 16 -nn 64 -b 128 -bits 5 -iter 1000 -channels 3 -snapshot snapshot
```

# Experiments

## t-SNE

```
cd run/experiments
python3 t_sne.py -snapshot ../snapshot
```

![scatter.png](https://qiita-image-store.s3.amazonaws.com/0/109322/0cbf77ad-5b71-adfe-a940-eaa24a81890f.png)

## Interpolation

```
cd run/experiments
python3 interpolation.py -snapshot ../snapshot
```

![interpolation.png](https://qiita-image-store.s3.amazonaws.com/0/109322/f4dd093c-c8c5-9759-ffbd-26e83286a76b.png)

## Generate images

```
cd run/experiments
python3 generate.py -snapshot ../snapshot
```

![mnist.gif](https://qiita-image-store.s3.amazonaws.com/0/109322/2294b41b-1f69-e88e-d80f-08e4fbbedf8d.gif)