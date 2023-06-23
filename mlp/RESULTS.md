# MLP: Git Re-Basin results

These are results from running `PermutationCoordinateDescent` and `MergeMany` on MLPs.

**Table of Contents**

- [Results](#results)
  - [PermutationCoordinateDescent](#permutationcoordinatedescent)
  - [MergeMany](#mergemany)
    - [By number of models](#by-number-of-models)
    - [By feature size](#by-feature-size)
    - [Larger sweeps](#larger-sweeps)
      - [Analyzing outputs](#analyzing-outputs)
      - [Analyzing the weights](#analyzing-the-weights)
- [The model](#the-model)
- [The permutations](#the-permutations)

# Results

Below is an analysis of the results of running 
`PermutationCoordinateDescent` and `MergeMany` on the MLP presented 
[here](#the-model).

## PermutationCoordinateDescent

I wanted to find out if `PermutationCoordinateDescent` works better with a higher L2-regularizer.
This would make sense, because then, the weights would be more similar in magnitude,
and therefore, more similar. Permuting them should work better.

Here are the results for an MLP (definition see [mlp_tests.py](mlp_tests.py))
with different settings for `weight_decay` in the `Adam` optimizer, 
which sets the L2-regularizer for the weights:

<p align="center">
    <img
        src="results/permutation-coordinate-descent/pcd_results.png"
        alt="PermutationCoordinateDescent results for MLP"
        width="600"
    />
</p>

And lo and behold, it works! The higher the L2-regularizer, the better the results, up to a point.

To be clear, better here means that the loss barrier is lower, not that the absolute loss is lower.
It's not, maybe because I've barely tuned the hyperparameters, and when I did, I did it 
without regularization, or it's because the `weight_decay` does get *very* high in some of these plots.

Some observations:

- At a `weight_decay` of 0.1, there is no longer a loss barrier. Increasing the `weight_decay` further
  doesn't seem to help.
- At the same time, the losses of the other interpolations have a strange dip in the middle, which
  is consistent over all further `weight_decay` values. I don't claim to know why this happens.
- It seems that for this model, interpolating between `model_a` and `model_b (rebasin)` yields
  higher-accuracy models than either of the original ones, if the `weight_decay` is high enough.
- Strange things happen to the accuracies at high `weight_decay` values. I don't know why.
- When the `weight_decay` is too high, a loss barrier start appearing again.

(Accuracy is top-1)

This is a clear success!!!

## MergeMany

The above makes it likely that `MergeMany` also works better with a higher L2-regularizer.

Here are the results:

<p align="center">
    <img
        src="results/merge-many/merge_many_wd_0.0-0.1.png"
        alt="MergeMany results for MLP"
        width="600"
    />
</p>

And for higher `weight_decay` values:

<p align="center">
    <img
        src="results/merge-many/merge_many_wd_0.0-0.9.png"
        alt="MergeMany results for MLP"
        width="600"
    />
</p>

The look promising, so let's normalize them by dividing by the loss / accuracy of the merged model 
at every step:

<p align="center">
    <img
        src="results/merge-many/merge_many_wd_0.0-0.1_norm.png"
        alt="MergeMany results for MLP"
        width="600"
    />
</p>

<p align="center">
    <img
        src="results/merge-many/merge_many_wd_0.0-0.9_norm.png"
        alt="MergeMany results for MLP"
        width="600"
    />
</p>

Clearly, with high `weight_decay`-values, `MergeMany` works pretty well; 
the loss is similar to that of the control model, and the accuracy is even higher.

### By number of models

While `weight_decay` isn't mentioned as a factor influencing the 
results of `PermutationCoordinateDescent` and `MergeMany`
in the paper, it is clearly important. 

Another factor, however, is the number of models that are merged.
Ideally, more models would mean better results. However, in practice, the opposite is true.
At best, high `weight_decay` values are needed to get good results with many models,
though they are still not better than with fewer models.

<p align="center">
    <img
        src="results/merge-many/merge_many_nm2-6_wd0-0.3_nf200.png"
        alt="MergeMany results for MLP"
        width="800"
    />
</p>

Observations:

- At low `weight_decay` values, `MergeMany` works significantly better with fewer models.
- At high `weight_decay` values, this effect almost disappears in the loss, 
   and seems to fully reduce the effect on accuracy &mdash; though that might be because
   the accuracy is a highly non-linear measure, making the data noisy.

### By feature-size

In the paper, `weight_decay` isn't mentioned as a factor 
influencing the results of `PermutationCoordinateDescent` and `MergeMany`.
On the other hand, feature-size is. So now, let's try `MergeMany` 
with different feature-sizes plotted over different `weight_decay` values.
The y-axis shows the ratio of the loss or accuracy of the merged model
to the loss or accuracy of the control model.

<p align="center">
    <img
        src="results/merge-many/full_wd0.0-0.9_hf100-2000_sweep.png"
        alt="MergeMany results for MLP"
        width="800"
    />
</p>

Here, `avg` stands for the average of the three models from before they are merged,
and `merged` stands for the merged model.

What becomes clear is that while the feature-size is very important at low `weight_decay` values,
it becomes almost irrelevant at higher `weight_decay` values. The `weight_decay`-value, 
on the other hand, is very important for `MergeMany`, irrespective of the feature-size
(though at higher feature-size, it isn't quite as important as at lower feature-size).

Let's take a closer look at lower `weight_decay` values:

<p align="center">
    <img
        src="results/merge-many/full_wd0.0-0.2_hf100-2000_sweep.png"
        alt="MergeMany results for MLP"
        width="800"
    />
</p>

Even a pretty low `weight_decay` appears to help `MergeMany` a lot.

Interestingly, the merged model never reaches a loss as low as the average of the models 
that make it up, but its top-1 accuracy is often higher. 
My hypothesis for why that is, is the following:

The merged model acts as an Ensemble, but the outputs of the Ensemble are averaged,
meaning that the outputs should come closer to their mean. This would express itself
in two things: 1.) The maximum output would be lower, and 2.) so would the standard deviation.
This would result in more uniform outputs, and thus a higher loss, 
but the results might still be better (as long as the information in the outputs
isn't averaged out into noise), resulting in a higher accuracy.

This view is supported by the following measurements. These show the ratio of 
some metric from the merged model to the same metric from the average of the models
that make it up. The metrics are accuracy, loss, maximum output, and standard deviation
of the outputs.

<p align="center">
    <img
        src="results/merge-many/compare_output_statistics_wd0.0-0.2_hf400.png"
        alt="MergeMany results for MLP"
        width="600"
    />
</p>

It's clear that the maximum output and the standard deviation of the outputs
are lower in the merged model than in the models that make it up, while loss and
accuracy are higher. 

An exception to this is the case where `weight_decay == 0.0`; the behavior in this case
is probably explained by `PermutationCoordinateDescent` &mdash; which is used
by `MergeMany` &mdash; achieving a worse fit between the models. The permuted models are
averaged, so if they fit poorly, they will converge towards random noise. 

This view is supported by the number of permutations that are applied to the models
before they are merged, at different `weigth_decay` values:

<p align="center">
    <img
        src="results/merge-many/count_permutations.png"
        alt="MergeMany: # Permutations for MLP over weight_decay"
        width="800"
    />
</p>

The number of permutations rises quickly at low `weight_decay`-values,
but then almost levels off. Clearly, an L2-regulizer helps `PermutationCoordinateDescent`
and, with that, `MergeMany`. It should be noted, however, that the difference in 
permutation counts isn't very large in relative terms. The lower two plots
show the same data, but with the y-axis scaled to start from 0, and it 
clearly shows this.

### Larger Sweeps

To find answers, let's do larger hyperparameter sweeps!

#### Analyzing outputs

This first image shows four outputs over `weight_decay` and `feature_size`:

1. The ratio of the loss of the merged model to the loss of the average of the models
   that make it up. (lower is better)
2. The ratio of the accuracy of the merged model to the accuracy of the average of the models
   that make it up. (higher is better)
3. The ratio of the maximum output of the merged model to the maximum output of the average of the models
   that make it up. (higher is better, because it means that the merged model is more confident)
4. The ratio of the standard deviation of the outputs of the merged model to the standard deviation
   of the outputs of the average of the models that make it up. (higher is better, because it means
   that the merged model is more confident)

<p align="center">
    <img
        src="results/merge-many/compare_output_statistics_wd0.0-0.2_hf50-1000_nm2-6_heatmap.png"
        alt="MergeMany: Summary"
        width="800"
    />
</p>

Let's analyze this:

- At low `weight_decay`-values, the std- and max-ratios are very low,
    while the loss and accuracy ratios are very high. 
    This makes me think that the reason that `weight_decay` is so important
    for `MergeMany` is that it makes the models more confident, and thus
    the averaging done in `MergeMany` doesn't average out the information
    in the outputs into noise, at least not as much.
- The feature-size, on the other hand, barely has an impact on this.
    Higher feature-size slightly increases the std- and max-ratios,
    but it's not very significant.
- The std- and max-ratios seem to worsen with the number of models that are merged.
    This is probably because the averaging done in `MergeMany` averages out
    the information in the outputs into noise, and the more models are merged,
    the more averaging is done.
    This effect is, however, reduced significantly at higher `weight_decay`-values.
- The fact that the std-ratio looks almost identical to the accuracy-ratio 
    seems to be more evidence for the hypothesis that the reason that `weight_decay`
    is so important for `MergeMany` is that it makes the models more confident.
- High `weight-decay`-values improve the accuracy of the merged model
    as compared to the average of the models that make it up.
    It seems that at a larger number of models that are merged,
    this happens more reliably (as evidenced by the less noisy image)

#### Analyzing the weights

In the following image, four values are plotted over `weight_decay` and `feature_size`:

1. The mean value of the elements of each weight.
2. The mean difference between the magnitude of the elements of the same weights in two models, 
    over all the weights in these models, in percent.
3. The mean difference between the mean eigenvalue of the weights in two models, 
    over all the weights in these models, in percent.
    This should, of course, correspond directly to the previous value.
    I did, however, calculate both to make sure that I didn't make a mistake.
4. The mean angle between a weight's eigenvectors in degrees.
    Measuring this values was a pot-shot.

<p align="center">
    <img
        src="results/other/weight_statistics_hf20-1000_wd0.0-1.0_nm2-2.png"
        alt="MergeMany: Summary"
        width="800"
    />

Let's analyze this:

- The mean value of the elements decreases with 
    both increasing feature-size and `weight_decay`.
    This is as expected. 
    Weights in `nn.Linear`-layers are initialized using
    `torch.nn.init.kaiming_uniform_`, which draws from a uniform distribution
    which has a range that decreases with increasing input-feature size.
    The `weight_decay`-term in the loss-function also decreases the weights,
    because it is a L2-regularizer.
- Both the percentage variation in the magnitude of the weights and the percentage variation
    in the mean eigenvalue of the weights increase with both increasing `weight_decay`
    and increasing feature-size.
    This might be an explanation for why both improve the results of `MergeMany`.
    As described [above](#analyzing-outputs), `weight_decay` makes the models more confident,
    and this might be because there is larger variation in the weight-values 
    of the models that are merged. 
    This might offset the fact that in `MergeMany`, 
    the weights of the merged models are averaged after being re-basined.
    In other words, hight `weight_decay`-values and feature-sizes
    might make the merged model more confident because they allow for 
    some variation in the magnitudes of the elements of the weights.
- The average angle between the eigenvectors of the weights is always around 90°...
    except for low feature-sizes at medium to high `weight_decay`-values.
    I have no idea why this is the case or what it means, but will attempt to analyze it further.


### Summary

- `weight_decay` is very important for `MergeMany`, and should be set to a high value.
- The feature-size is important for `MergeMany` at low `weight_decay` values, but
  becomes less important at higher `weight_decay` values.
- The number of models used for merging negatively impacts the results of `MergeMany`
  at low `weight_decay` values, but becomes irrelevant at higher `weight_decay` values.
  (it might be interesting to check if it decreases bias, which is an important point
   of the paper)
- The merged model never reaches a loss as low as the average of the models that make it up,
  but its top-1 accuracy is often higher.
- The maximum output and the standard deviation of the outputs are lower in the merged model
  than in the models that make it up, while loss and accuracy are higher.
- The number of permutations rises quickly at low `weight_decay`-values,
  but then almost levels off.

# The model

<p align="center">
    <img
        src="results/graph_hf784.png"
        alt="MLP model"
        width="400"
    />
</p>

# The permutations

```
                                         
PathSequence(                            
-----------------------------------------
                                         
LinearPath(                              
  DefaultModule(                         
    module.type: Linear                  
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation: None      
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  InputPermIsOutputPermMultiDimModule(   
    module.type: LayerNorm               
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  DefaultModule(                         
    module.type: Linear                  
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  InputPermIsOutputPermMultiDimModule(   
    module.type: LayerNorm               
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  DefaultModule(                         
    module.type: Linear                  
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  InputPermIsOutputPermMultiDimModule(   
    module.type: LayerNorm               
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  DefaultModule(                         
    module.type: Linear                  
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  InputPermIsOutputPermMultiDimModule(   
    module.type: LayerNorm               
    input.shape: [(64, 784)]             
    output.shape: [(64, 784)]            
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation.shape: 784
  )                                      
                                         
  DefaultModule(                         
    module.type: Linear                  
    input.shape: [(64, 784)]             
    output.shape: [(64, 10)]             
    weight.in_dim.permutation.shape: 784 
    weight.out_dim.permutation: None     
  )                                      
                                         
)                                        
-----------------------------------------
)                                        
```
