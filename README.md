# :construction: Work in Progress :construction:

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