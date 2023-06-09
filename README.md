# rebasin-results
Results for [snimu/rebasin](https://github.com/snimu/rebasin),
a library that implements methods described in 
["Git Re-basin"-paper by Ainsworth et al.](https://arxiv.org/abs/2209.04836)
for (almost) arbitrary models.

For acknowledgements, see [here](https://github.com/snimu/rebasin#acknowledgements).
For terminology, see [here](https://github.com/snimu/rebasin#terminology).


## MLP

Results for a simple MLP. These are very positive!
The loss barrier was successfully removed in `PermutationCoordinateDescent`,
and `MergeMany` worked well, too.

For detailed results, read [mlp/RESULTS.md](mlp/RESULTS.md).

Here's a taste:

<p align="center">
  <img 
    src="mlp/results/permutation-coordinate-descent/pcd_results.png" 
    alt="Losses" 
    width="600"
  />
</p>

The L2-regularizer (`weight_decay`) improves the performance of `PermutationCoordinateDescent`!

In `MergeMany`, I found that a high `weight_decay` is similarly important, 
though here I also looked at the effect of the feature size:

<p align="center">
  <img 
    src="mlp/results/merge-many/full_wd0.0-0.2_hf100-2000_sweep.png" 
    alt="MergeMany results" 
    width="800"
  />
</p>

Clearly, a higher feature-size is beneficial, but the effect of the weight-decay
is stronger.


## hlb-CIFAR10

Results from forking [tysam-code/hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10).
See [snimu/hlb-CIFAR10](https://github.com/snimu/hlb-CIFAR10) for the fork.

For results, read [RESULTS.md](hlb-CIFAR10/RESULTS.md). 
Raw data at [hlb-CIFAR10/results](hlb-CIFAR10/results).

**Summary**:

- `PermutationCoordinateDescent` seems to work fairly well, though not perfectly.
- `MergeMany` doesn't work as described in the paper, but when training the models
    that are merged on different datasets and then retraining on yet another dataset,
    the results are promising.

## hlb-gpt

Results from forking [tysam-code/hlb-gpt](https://github.com/tysam-code/hlb-gpt).
See [snimu/hlb-gpt](https://github.com/snimu/hlb-gpt) for the fork.

There is currently still some error with the code, so the results are not
meaningful. I have therefore not written a RESULTS.md yet.

If you would like to look at the raw data from an experiment, though, 
you can, at [hlb-gpt/results](hlb-gpt). I just want to repeat that this data
is produced by erroneous code, otherwise `model_b_original` and `model_b_rebasin`
should have the same loss and accuracy, at least approximately.

## torchvision.models.mobilenet_v3_large

Results for torchvision.models.mobilenet_v3_large.

For results, read [RESULTS.md](torchvision-models/mobilenet_v3_large/RESULTS.md).

**Summary**:

This model contains several `BatchNorm2d` layers. Those have to have their
running_stats recomputed after re-basing. This, however, takes a lot of compute
on ImageNet, for which I don't have the budget. What I can say is that 
as I increase the percentage of the ImageNet-data used for recalculating the 
running_stats, the results improve, so if someone would like to 
recompute the running_stats using more data (ideally all of imagenet),
I would be very interested in the results.

If you do want to run that test, look at 
[rebasin/tests/apply_to_real_models](https://github.com/snimu/rebasin/blob/main/tests/apply_to_real_models.py)
and run it with:

```bash
python apply_to_real_models.py -m mobilenet_v3_large -d imagenet -s 99 -v
```

Make sure to install the requirements first.
