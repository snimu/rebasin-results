# hlb-CIFAR10: Git re-basin results

These are the results from forking [tysam-code/hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10).
See [snimu/hlb-CIFAR10](https://github.com/snimu/hlb-CIFAR10) for the fork.

Below are first the losses and accuracies plotted for the model, then a graph of the model itself
(generated using [torchview](https://github.com/mert-kurttutan/torchview)).

## Results

The first plot shows the results
for the original model, while the others show results for ever-increasing filter-sizes of the 
convolutions. 
It is mentioned in the paper that the results are better for larger filter-sizes.

### Filter-size: 3x3

The original filter size.

<p align="center">
    <img
        src="3x3-plot.png" 
        alt="Results of Git re-basin for hlb-CIFAR10 with filter-size 3x3"
        width="600"
    />
</p>

### Filter-size: 6x6

<p align="center">
    <img
        src="6x6-plot.png" 
        alt="Results of Git re-basin for hlb-CIFAR10 with filter-size 6x6"
        width="600"
    />
</p>

### Filter-size:9x9

<p align="center">
    <img
        src="9x9-plot.png" 
        alt="Results of Git re-basin for hlb-CIFAR10 with filter-size 9x9"
        width="600"
    />
</p>

### Filter-size: 12x12

<p align="center">
    <img
        src="12x12-plot.png" 
        alt="Results of Git re-basin for hlb-CIFAR10 with filter-size 12x12"
        width="600"
    />
</p>

### Filter-size: 15x15

<p align="center">
    <img
        src="15x15-plot.png" 
        alt="Results of Git re-basin for hlb-CIFAR10 with filter-size 15x15"
        width="600"
    />
</p>

### Filter-size: 18x18

<p align="center">
    <img
        src="18x18-plot.png" 
        alt="Results of Git re-basin for hlb-CIFAR10 with filter-size 18x18"
        width="600"
    />
</p>

### Analysis

A few things immediately jump out to me from the plots above:

1. The method works somewhat; interpolation between `model_a` and `model_b (rebasin)`
    is much better than between `model_a` and `model_b (original)`.
2. Applying the method to `model_b` and then interpolating between 
     `model_b (original)` and `model_b (rebasin)` yields better results than
     interpolating between `model_a` and `model_b (original)`. 
     This is not fully unexpected, because these two are of course fairly
     close to each other, but it's also not obvious (at least not to me).
3. The git re-basin method works very well for the accuracy of the model!
     At least for this model, interpolation between `model_a` and `model_b (rebasin)`
     leads to almost flat accuracies. 
4. Larger filter size is said to work better in the paper, 
    but it is unclear to me if this is actually the case here. Let's look at that 
    in more detail below.

#### Filter-size analysis

Below, I plot the losses and accuracies 
when interpolating between `model_a` and `model_b (rebasin)`.

I do so for all filter-sizes.

<p align="center">
    <img
        src="losses-all.png" 
        alt="Losses of Git re-basin for hlb-CIFAR10 with different filter-sizes"
        width="600"
    />
</p>

<p align="center">
    <img
        src="accuracies-all.png" 
        alt="Accuracies of Git re-basin for hlb-CIFAR10 with different filter-sizes"
        width="600"
    />
</p>

It is not clear to me that larger filter sizes lead to better results. 
However, larger filter-sizes do degrade model-performance in general
(the larger the filter size, the worse the performance of the first and last model,
which are the original `model_a` and `model_b (rebasin)`), 
so to give a comparison of how rebasin affects how interpolation between models
behaves, I plot the losses and accuracies again below, but this time 
I move all startpoints (i.e. `model_a`) to the results of the 3x3-filter.

<p align="center">
    <img
        src="losses-all-normalized-startpoint.png" 
        alt="Losses of Git re-basin for hlb-CIFAR10 with different filter-sizes"
        width="600"
    />
</p>

<p align="center">
    <img
        src="accuracies-all-normalized-startpoint.png" 
        alt="Accuracies of Git re-basin for hlb-CIFAR10 with different filter-sizes"
        width="600"
    />
</p>

The behavior seems noisy, though slightly better for larger filter-sizes.

## Model

The model is a simple Resnet:

<p align="center">
    <img
        src="hlb-cifar10-model-graph.png" 
        alt="hlb-CIFAR10 model graph"
        width="400"
    />
</p>

## Permutations

The model with 3x3-filters has the following permutation structure:

```
PathSequence(                                                  
----------------------------------------                       
                                                               
LinearPath(                                                    
  DefaultModule(                                               
    module.type: Conv                                          
    input.shape: [(5000, 3, 32, 32)]                           
    output.shape: [(5000, 12, 31, 31)]                         
    weight.in_dim.permutation: None                            
    weight.out_dim.permutation.shape: 12                       
  )                                                            
                                                               
  DefaultModule(                                               
    module.type: Conv                                          
    input.shape: [(5000, 12, 31, 31)]                          
    output.shape: [(5000, 32, 31, 31)]                         
    weight.in_dim.permutation.shape: 12                        
    weight.out_dim.permutation.shape: 32                       
  )                                                            
                                                               
  DefaultModule(                                               
    module.type: Conv                                          
    input.shape: [(5000, 32, 31, 31)]                          
    output.shape: [(5000, 64, 31, 31)]                         
    weight.in_dim.permutation.shape: 32                        
    weight.out_dim.permutation.shape: 64                       
  )                                                            
                                                               
  OneDimModule(                                                
    module.type: BatchNorm                                     
    input.shape: [(5000, 64, 15, 15)]                          
    output.shape: [(5000, 64, 15, 15)]                         
    weight.in_dim.permutation.shape: 64                        
    weight.out_dim.permutation.shape: 64                       
  )                                                            
                                                               
)                                                              
----------------------------------------                       
                              |                                
                              |                                
                              |                                
-------------------------------------------------------------- 
ParallelPaths(                                                 
  LinearPath(                                 LinearPath()     
    DefaultModule(                                 |           
      module.type: Conv                            |           
      input.shape: [(5000, 64, 15, 15)]            |           
      output.shape: [(5000, 64, 15, 15)]           |           
      weight.in_dim.permutation.shape: 64          |           
      weight.out_dim.permutation.shape: 64         |           
    )                                              |           
                                                   |           
    OneDimModule(                                  |           
      module.type: BatchNorm                       |           
      input.shape: [(5000, 64, 15, 15)]            |           
      output.shape: [(5000, 64, 15, 15)]           |           
      weight.in_dim.permutation.shape: 64          |           
      weight.out_dim.permutation.shape: 64         |           
    )                                              |           
                                                   |           
  )                                                |           
)                                                              
-------------------------------------------------------------- 
                   |                                           
                   |                                           
                   |                                           
-----------------------------------------                      
                                                               
LinearPath(                                                    
  DefaultModule(                                               
    module.type: Conv                                          
    input.shape: [(5000, 64, 15, 15)]                          
    output.shape: [(5000, 256, 15, 15)]                        
    weight.in_dim.permutation.shape: 64                        
    weight.out_dim.permutation.shape: 256                      
  )                                                            
                                                               
  OneDimModule(                                                
    module.type: BatchNorm                                     
    input.shape: [(5000, 256, 7, 7)]                           
    output.shape: [(5000, 256, 7, 7)]                          
    weight.in_dim.permutation.shape: 256                       
    weight.out_dim.permutation.shape: 256                      
  )                                                            
                                                               
)                                                              
-----------------------------------------                      
                              |                                
                              |                                
                              |                                
---------------------------------------------------------------
ParallelPaths(                                                 
  LinearPath(                                  LinearPath()    
    DefaultModule(                                  |          
      module.type: Conv                             |          
      input.shape: [(5000, 256, 7, 7)]              |          
      output.shape: [(5000, 256, 7, 7)]             |          
      weight.in_dim.permutation.shape: 256          |          
      weight.out_dim.permutation.shape: 256         |          
    )                                               |          
                                                    |          
    OneDimModule(                                   |          
      module.type: BatchNorm                        |          
      input.shape: [(5000, 256, 7, 7)]              |          
      output.shape: [(5000, 256, 7, 7)]             |          
      weight.in_dim.permutation.shape: 256          |          
      weight.out_dim.permutation.shape: 256         |          
    )                                               |          
                                                    |          
  )                                                 |          
)                                                              
---------------------------------------------------------------
                   |                                           
                   |                                           
                   |                                           
----------------------------------------                       
                                                               
LinearPath(                                                    
  DefaultModule(                                               
    module.type: Conv                                          
    input.shape: [(5000, 256, 7, 7)]                           
    output.shape: [(5000, 512, 7, 7)]                          
    weight.in_dim.permutation.shape: 256                       
    weight.out_dim.permutation: None                           
  )                                                            
                                                               
  OneDimModule(                                                
    module.type: BatchNorm                                     
    input.shape: [(5000, 512, 3, 3)]                           
    output.shape: [(5000, 512, 3, 3)]                          
    weight.in_dim.permutation: None                            
    weight.out_dim.permutation: None                           
  )                                                            
                                                               
)                                                              
----------------------------------------                       
                            |                                  
                            |                                  
                            |                                  
-----------------------------------------------------------    
ParallelPaths(                                                 
  LinearPath(                              LinearPath()        
    DefaultModule(                              |              
      module.type: Conv                         |              
      input.shape: [(5000, 512, 3, 3)]          |              
      output.shape: [(5000, 512, 3, 3)]         |              
      weight.in_dim.permutation: None           |              
      weight.out_dim.permutation: None          |              
    )                                           |              
                                                |              
    OneDimModule(                               |              
      module.type: BatchNorm                    |              
      input.shape: [(5000, 512, 3, 3)]          |              
      output.shape: [(5000, 512, 3, 3)]         |              
      weight.in_dim.permutation: None           |              
      weight.out_dim.permutation: None          |              
    )                                           |              
                                                |              
  )                                             |              
)                                                              
-----------------------------------------------------------    
                 |                                             
                 |                                             
                 |                                             
------------------------------------                           
                                                               
LinearPath(                                                    
  DefaultModule(                                               
    module.type: Linear                                        
    input.shape: [(5000, 512)]                                 
    output.shape: [(5000, 10)]                                 
    weight.in_dim.permutation: None                            
    weight.out_dim.permutation: None                           
  )                                                            
                                                               
)                                                              
------------------------------------                           
)   
```

The models with larger filter-sizes are exactly the same, 
except for the convolutions' weight-shapes.
