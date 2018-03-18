# Feature Pyramid Networks For Object Detection

This paper aims to provide a fast way to construct feature pyramids for different scales of an image. Previous pyramid representations took up a lot of time/space to compute
Here a Feature Pyramid Network is constructed which takes into account the intrinsic nature of conv nets to construct feature pyramids at marginal cost.


# Author 
Facebook AI (FAIR) 

## Paper Link
https://arxiv.org/pdf/1612.03144.pdf

# Nomenclature

```
@ -> Matrix Multiplication
* -> Hammard's Product
```

## Introduction

1. This model unlike previous models , creates a feature pyramid that combines ```low resolution, semantically strong features with
high resolution, semantically weak features``` using ```top down pathway and lateral connections.```

2. Previous Models like SSD , used last layers to compute feature pyramids hence skipping high resolution maps, thereby decreasing accuracy in detecting
small objects

3. Aim is to create an in network feature pyramid, hence saving on time/space

4. The authors incorporated a FPN in Faster R-CNN and achieved state of the art results

5. Hence network can be trained at all scales and is used consistently at test/train time.

# Model

1. Resnet used to construct feature maps.

2. There are two things, bottom up and top level pathways
    * Bottom Up Pathways are the *feature maps* after each residual block (in the Resnet) , hence denote features at different depths
    * Top Level Pathways are upsampled features from higher lvele smaller feature maps. Upsampling is done with a factor of two using (nearest neighbour upsampling)

## Combining these layers

1. These two top level and bottom up layers of the same size are combined using addition. Hence is the cae of resnet you will have a final 4 layers. Conv1 is not used aas it takes up a lot of memory

2. Bottom up layers are undergone a 1 x 1 conv to bring them to the same channel dimensions.

3. The final layers are made to undergo a further 3 x 3 conv . Hence getting P2,P3,P4,P5.

4. One channel dimension is used that is 256.

# Modifications to existing Networks to Incorporate FPN's

## FPN for RPN (Region Proposal Network)
Todo , need to read RPN paper first



