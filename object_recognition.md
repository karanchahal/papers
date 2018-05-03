# Survey Of Object Recognition

The topics are as follows:

1. 2 Step Approaaches
2. Single Shot Approaches

2 step approaches can talk about R-CNN, Fast R-CNN, Faster RCNN.
Then talk about innovations of each, slowly moving towards a fully convolutional model. Roi Pooling updated to R-FCN.

Meanwhile, YOLO , SSD providing a new way of doing things.

Also talk about convolutional bases and how they've evolved, inspiration from NLP solutions.

Finally newest solution is Retina Net, mask R-CNN. 

Newest verison of Yolo (v3) out. 

Tabulation of scores.

Author's perspective as to where the field is eventually going. The need for real time object recognition .

Faster,  more accurate. 

The roadblocks, class imbalance etc.

Need for faster detectors

Training Details.

1.State of the art training (Distributed SGD), model distillation (?)

2.Binary Networks, Mobile Nets, Shuffle Nets to provide very efficient conv bases for running on mobile applications

3.AutoML Advancements in Future Work. 

# Introduction

It all started with 2012 when ALex Krchevsky upublished the AlexNet paper , which attained breakthrough performances in the Image Net challenge for image classification. It was the first time deep neural networks had proved to work amazingly.

After the breakout of deep learning, various models came out using neural networks as their core. Speech Recogniotion, NLP , Reinforcement Learning , were all affected by deep learning.

Similiarly , people took to apply these deep nets to detect objects in images. 
Object detection is the task of finding objects in an image. Their location of an object in an image and what type of object it is. 

Most image tasks were being solved with deep convolutional nets, the very same innovation in Alex Krochevsky's paper.

The image went through the conv nets to provide a feature map, which was smaller in width and height but was deeper in channels. It provide an alterate representation of an image. 

This feature map was further worked upon , to identify objects an their locations.

The first model that had good results was the R-CNN or the Region Convolutional Network.

# Region Convolutional Network

In a R-CNN,  

There were 2 steps to this model 
1. Find good proposals or regionsin an image where an object could exist
2. Train on these proposals, to predict objects

So firstly the model should get good proposals/ regions of interest. This was done with a non deep learning algorithm called selective search. Selective search got x amount of proposals or blobs of an image, then those blobs were cropped out and resized . Then theese resized blobs were fed into the RCNN network,to predict bounding boxes and classes of objects.


### Drawbacks

The RCNN neytwork has several drawbacks 


1.  It had a complex traning procedure.Training is a multi-stage pipeline.There are 3 stages , fintuning feature map with log loss. SVM CLassiciation head to be trained to classifiy objects and infally object regression heads are trained to ouput coorindates of bounding boxes

2. Training is expensive in space and time. For SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk. With very deep networks, such as VGG16, this process takes 2.5 GPU-days for the 5k images of the VOC07 trainval set. These features require hundreds of gigabytes of storage.

3. It was slow, it took approximately 47 seconds to process a singgle image. The bottleneck as observed to be selective search which took 27 seconds on its own.

# Fast RCNN

To address these concerns, a new version of RCNN called Fast RCNN was made. The major innovation of Fast RCNN was that, it shared the features of a convolutional net and constructed a single step pipeline which trained a multitask loss. 

The step in R-CNN  which cropped each of the object proposals from the original image and run each of them through the conv net, was innovated upon mainly.

The innovation was a method called ROI Pooling

ROI Pooling or Region Of Interest Pooling is as follows 

1. Cropped the ROI's from the feature map itself instead of the input image, thus sharing computation. 

2. Also after these ROI's were cropped out of a feature map and resized to a fixed size (7 by 7) , It was run though the Fast RCNN model


The Fast RCNN Network was made of a feature map, followed by a few convolutional/pooling layers. 
Then two prediction heads were attached on the final feature map.

1. A classification head (to predict the types/class of objects) 
2. A regression/location head ( to predict coordinates of objects)

3. This was trained using a multitask loss which was 

* Loss_of_classification + lambda(loss of regression)

* lambda switched off and on if the region were background or not respectively.

Hence this sharing of computation resulted in 

1. Higher accuracy
2. Speed and Space efficiency (Testing on a single image reduced from 47 seconds to .32 seconds)
3. Training time reduced substantially ( from 85 hours to 9 hours)

### Drawbacks

Even though Fast RCNN was a huge improvement, still some drawbacks were consistent.

1. Bottleneck to the speed was due to the selective search procedure which took up a huge percentage of the time taken.
2. Achieved near real-time rates using very deep networks _when ignoring the time spent on region proposals_. Now, proposals were the test-time computational bottleneck.

# Faster RCNN

To solve the above drawbacks , the authors decided to iterate RCNN framework one more time.

They had identified that the calculation of proposals took a lot of time, so they decided to replace this method with a better method.

Hence they set out to learn how to calculate proposals by using a neural network , making it signicifantly faster to calculate regions of interests and maybe making it more accurate.

So a major ,milestone in object recognition was achieved called the *Region Proposal Network* . 
This network learns to calculate good quality regions of interest/object proposals.

The model is as follows:

1. Taking an input image.

2. Targets are the data the model is trained against. 

3. Targets are calculated taking the ground truth bounding boxes and calculating some regions of interest from that data. 

We use these targets to train against, and not the ground target boxes because of : 
Intuitively we can think of it , as we are making the learning task a litle easier. Instead of predicting the final boxes, we tell the network to predict a large number of plausible boxes.

These target regions proposals are generated with a concept known as anchors.

What are anchors? Anchors are , simply put, predefined crops of the input image (224 by 224 )

More specifically they are patches(of size axb) cropped after x number of pixels in the image. The size of the patches is described below.

In Faster RCNN, 9 types of sizes are used. These sizes are decided upon using a concept of aspect ratios of images and scales . Example: Aspect Ratios =>  1:1, 1:2, 2:1 and Scales  =>  32px, 64px and 128px. Hence 9 shapes.

### Second Major Point

These shapes are cropped after every 16 pixels in Faster RCNN

We arrive at a number 16 because the VGG Net constructs a feature map which is of 

1. Width = Image_width/16
2. Height = Image_height/16

Hence,the reason for this is we want every grid cell or box in the feature map to represent anchors . And each grid cell position , 9 anchors are cut.  

Think of the grid cell as the center pixel point of all these 9 anchors.

* Thus the feature map of 7 by 7 with a depth dimension of 9 will represent all anchors possible in the image

* Now, we need two target values for each anchor.
 1.We want to predict what class that anchor is. Is it a dog, cat etc.
 2. And we also want to predict what needs to be the modification to the shape of the anchor box to better fit the object. For this we need offsets of the anchor height, anchor width and anchor x and anchor y coordinates.

### Calculation of Targets

1. If the anchor has an IOU > 0.7 with a ground truth box, we assign it the class of that ground truth. We also calculated the error coordinates values.

2. IF IOU is < 0.3 , then class is background.

3. Other anchors are ignored

Hence target calculation is done for each and every anchor. 

Now once we have the targets, 

1. only 256 anchors are used to train an RPN in a single batch.
2. Half of them are background class anchords and other half are foreground.

Now we have the targets ready , to train on this data we use the following structure

### RPN STRUCTURE

1. Apply 3 by 3 conv map with padding 1, to retain same image height and width.

2. Two prediction heads, classification and regression. using 1 by 1 convolution but changing channel depth to 9 (types of anchors for every pixel position)

Softmax + log loss is used for the classification head and Regression loss is calculated using Smooth L1 loss. (description needed ?)

Again, similiar to Fast R-CNN

3. This was trained using a multitask loss which was 

* Loss_of_classification + lambda(loss of regression)
* lambda switched off and on if the region were background or not respectively.


Now after we have arrived at finding good object proposals, we can continue on with the other Fast RCNN steps of

1. ROI Pooling and then
2. The classification network/regression network ( which is very similiar to RPN ) to find final bounding boxes.


Hence we have achieved a full 2 step pipeline that is end to end and fully learnt. This paper was a fundamental breakthrough in deep learning.

### Note

* One question that may come to mind is that why do we need to compute proposals .
* Why can't be train on all the anchors and use all of them as proposals ?

The reality is that,  that leads to the model not being able to learn as there are way more background anchors than foreground anchors . Leading to the dataset being skewed in the favour of background classes and the model not being able to learn the objects well. 

It would simply predict background for all anchors .


# R-FCN Region Fully Convolutional Network

Making the object detection fully convolutional, hence quite a bit faster, but retaining accuracy. 

1. So the major improvement of the R_FCN is its attempt to make the whole model fully convolutional.

2. Previosly models had a cropping of the feature map into 7 by 7 and then running that region through a 3by3 conv and 2 1 by 1 convs

3. This lead to substantial time being spent to do that for each ROI

4. In the new model the feature map is underwent through a 1 by 1 conv , therby increasing its dept to k*k*(c+1) size. Where k can be any number, (in the paper k = 7) . c are the number of classes. c+1 because of background class. They try to deal with the problem of balancing traslation variance and invariance.

5. Then this model is taking this feature map. Softmax is calculated directly. So no extra conv layers, leading to better speeds.

6. Byut how does this softmax work. It a modified version of softmax.

7. Here , the ROI is divided into k by k bins. And then the values of each bin ,is calculated by its corresponding feature map (k*k*(c+1)). Hence , this feature map is converted into a c+1 vector. (max/averagepooling on 2 steps. 1st for getting k into k values, then getting their average to get 1 value)

8. Softmax is computed on this and trained with the targets.

9. To calculate the bounding boxes, we take a feature map of depth k*k*4 hence getting a value of 4 to train the targets.
```
Meanwhile, our results
are achieved at a test-time speed of *170ms* per image using ResNet-101, which is 2.5× to 20× faster
than the Faster R-CNN + ResNet-101 counterpart.
```
```
These experiments demonstrate that this
method manages to address the dilemma between invariance/variance on translation, and fully convolutional
image-level classifiers such as ResNets can be effectively converted to fully convolutional object
detectors
```


# R-FPN Region Feature Pyramid Network

This paper, introduces an extension to the Faster RCNN framwork.

This extension aims to provde a way to train the model to be robust enough to be able to detect images that are of different sizes and scales. It incorporates a model mechanism that takes variations of images, depending on scale and gets predictions.


Region Feature Pyramid Network (FPN)



The Feature Pyramid Network (FPN) is a neural network which was constructed to deal with images of different scales. 

Traditionally computer vision algorithms dealt with images of multiple resolutions by constructing a feature pyramid network.

Feature pyramids are representations of an image at different scales. They are called a pyramid as the scale becomes smaller and smaller as we go up, hence the name feature pyramid.

This feature pyramid structure enables a model to detect objects across a large range of scales by scanning the model over both positions and pyramid levels.

Convolution nets have been found out to be quite robust to change in scales and thus good results have been achieved by training on a single scale network. But to achieve the best results, multi scale training was being done. 

Multi-Scale training is quite resource intensive and takes up to a minute to process an image. This was far from ideal for a real time solution, hence FPN’s . 


### The Main Idea

A feature pyramid network leverages the pyramidal structure of a convolutional network, and computes predictions on the various scales that the convolutional feature map goes through, to form the final feature map.

For example in the Resnet, the image goes through 6 scale changes. 224, 56, 28, 14, 7.

There have been approaches to leverage the pyramidal structure of convolutional nets to train a network for robust multi scale detection. But those approaches simply applied a prediction at each feature map. This approach did not consider the high level features that were formed in the smaller feature maps. 
Feature Pyramid Networks used these high level features to compute predictions , as well as build a feature pyramid.

At each scale , the following transformations take place.

```
The feature pyramid is run through a 1 by 1 conv, to bring the number of channels to 256.We call it A.
The feature map one level higher , is upsampled by a factor of 2 (nearest neighbour upsampling) , we call it B. This is called a top down pathway.
A and B are summed together element wise to form C. This is called a lateral connection.
A 3 by 3 conv net is run through C to get final feature map D.
```

D is computed for all 4 scales. Hence D1,D2,D3,D4. Predictions are done for all of these 4 feature maps.
Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway. The bottom-up feature map is of lower-level semantics, but its activations are more accurately localised as it was subsampled fewer times. There are no non linearities introduced in these conv nets.

The FPN network can be embedded into current object detector architectures very easily. Now instead of a single feature map, everything is computed keeping multiple feature maps in mind. 

Including FPN in Single Step Object Detectors. 

Architectures like Yolo, Retina Net, RPN can be extended with there FPN very easily.
 So instead of using a single feature map to run the two convolutional heads of Classification and regression. The heads are run on all the feature maps {D1,D2,D3,D4}

Because the heads slide densely over all locations in all pyramid levels, it is not necessary to have multi-scale anchors. Hence the number of anchors at kept to just the number of aspect ratios. 
{1:1,1:2, 2:2} are most common. 

The  area of anchors at each level are of area {16, 32, 64, 128, 256}. So in total there are 15 anchors over the pyramid. 

 The parameters of the heads are shared across all feature pyramid levels.This advantage is analogous to that of using a featurized image pyramid, where a common head classifier can be applied to features computed at any image scale.

In Fast RCNN network, the ROI is cropped into a 7 by 7 shape on the feature map level where the classification accuracy was the highest. Two fully connected MLP’s are run through all levels of the feature maps.

FPN’s have been proven to increase accuracies of  existing object detection models with no bells and whistles and have substantially  increased accuracy in detecting small objects.

```
Iterative regression : ?
Hard negative mining : finding key negative samples/ anchors (confusing) , then training on these negative samples. Increases examples.

Context modeling: ?
Stronger data augmentation in SSD ?
```

# Retina Net

While making these object detection models, single step models always lagged behind two step algorithms. It was understood that the two step models did a better job of finding out regions of interests, while the single step models did not .
The authors of RetinaNet pondered on why the single shot detectors were worse than the two step approaches. They suggested that they did not fit well because of class imbalance.
Class imbalance indicates that there is an imbalance between the  number of positive and negative training examples. Training examples were heavily skewed in the favour of the negative classes. This was because in the entirety of the image very few anchors had actual objects in it. Hence background anchors dominated the foreground anchors.
The two step approach did not have this problem because it found out an even mix of foreground/background anchors to train on with the help of the RPN network. Though even with the RPN, it was difficult to find at the same number of positive and negative anchors.
RetinaNet is a single step object detector neural network ,that brings changes to the loss function.
The authors came up with a novel loss function which modulates the effect of the loss depending on what class it is predicting. So the loss function checks that if it’s a negative class label, then reduces the loss if the model predicts a class very confidently. And maintains the magnitude of the loss if the model is not very confident in it’s prediction. 

The loss function is as follows:
```
pt = -(1-pt)^ylog(pt)
```

(1-pt) gives a high value if pt is not confident in the prediction and a low value if it is. The authors decided to lower the effect of the loss if the model is confident with the prediction. This was done so that the loss is not dominated by easy examples. the extra value y is used to modulate the effect of this down weighting. Generally, y=2 works well in practice.
-log(pt) is the normal cross entropy loss.

The Retina Net architecture is similar with the YOLO Network. The only difference is that, an FPN is added too.


# Convolutional Bases

-> Resnet
-> ResNeXt
-> Mobile Net (research efficient versions)

# YOLO

Yolo is an abbrivation for You Only Look Look Once. It's the first of Object Detection techniques mentioned here, wherein object detection and classification is encompassed in a single process.

The primary inspiration behind making YOLO was to make object detection and classification real time. This would require the whole proccess, from dection of potential objects to their classification to be completed significantly faster.
Other region proposed classification models are required to run prediction for each region proposal, which in itself is a time consuming . Elimination of this repitation could make the process much faster.
Yolo achieves the aforementioned goal of having to run the entire image through the prediction model only once.


## Process

1. The entire image is devided into a grid of SxS
2. Each of the grid cells, predicts B bounding boxes.
3. Each bounding box predicts 5 values: 4 spatial coordinates and Confidence of the object being present in the box. 
   Each grid also gives out the Class probability of each class. Therefore the predictions are encoded as a SxSx(B*5+C) tensor.

4. The class probabilities are Conditional probablities P(Class|Object).

## Network
The Network Architecture used in YOLO is inspired from the GoogleLeNet model for image classification.
1. YOLO has 24 convulutional layers, which are used for feature extraction.
2. 2 fully connected layers, used for the prediction of probabilities and coordinates.
3. Instead of the Inception Module, 1x1 reduction layer is used to reduce the feature space from preceding layer.
4. The final layer predicts the bounding boxes coordinates as well as class probabilities.
5. All layers, except the final layer, which uses a Linear Activation funtion, use Leaky Rectified Linear activation defined by the following -
```
f(x) = {x :x>0 , 0.1x :otherwise}
```

## Loss Function
Sum Squared Loss function loss function is used because of the relative ease with which it can be optamised. This loss function does come with bundled with the followign drawbacks-
1. It gives equal weightage to the errors associated with Classification and in finding the bounding the box coordinates(localisation).
2. Most of the grid cells don't have contain any object. This brings the Confidence score associated with them very close to zero, thus overpowering the gradient from cells that do in fact contain an object.

To resolve the above issues, YOLO increases the loss obtained from bounding box coordinates and decreases the loss from grid cells that don't contain any object in them. This is done using contant mutipliers of λc=5 and λno=0.5 to the loss associated with Bounding box coordinates and Objectless grid cells respectively.

3. The loss fucntion also gives equal importance to errors in small and big bounding boxes. Whereas it's in coherence with common logic that, a small error in a small box should be given importance over the same error in a much larger box.

The following multipart Loss Function is optamised over the enitre training procedure-

[eq for the funct]



## Training
1. Darknet framework is used for all training procedures.
2. 20 Convulutional, an average pooling layer and a fully connected layer are pretrained on the ImageNet 1000-class competition dataset.
3. The remaining 4 Conv layers along with the 2nd Fully connected layer are added to the model and finally trained to perform detection. 
4. The network was trained for about 135 epochs on the Pascal VOC 2007 dataset. 
5. A batch size of 64, momentum of 0.9 and decay of 0.0005 were used throughout the training.
6. For the first epochs the learning rate was slowly raised from 0.001 to 0.01. It was noticed that if the training started at a high learning rate, the model often diverged due to unstable gradients. Training was continued with 0.01 for 75 epochs, then 0.001 for 30 epochs, and finally 0.0001 for 30 epochs.


## Advantages
1. It's mush faster due to the ealy training pipeline.
2. It's able to more easily generalize objects.
3. It implicitley encodes the contextual info about classes and their appearance since it sees the full image. 

# Drawbacks
1. It has a much lower accuracy.
2. It performs poorly for smaller objects as the image is reduced to a SxS grid.
3. Since each grid can have atmost B bounding boxes, it performs poorly with many samll objects.
4. Loss function treats the errors in small bounding boxes and large boxes with the same weightage.
5. Input resolution is increased to 448x448 from 224x224.


# Yolo2

Yolo 2 made several improvements to Yolo 1
Key improvements are 

1. Increased Image Resolution. Boost from 224 to 448
2. Using anchor boxes, decoupling the class prediction and regression heads akak like Faster RCNN
3. Using K-means to find good anchor scales and aspect ratios.
4. A new way of predicting offsets to coordinates. Constrains to be closer to the anchor box.
5. A new classifier, that was deried from Googlenet, which was faster than the VGG counterpart.
6. More fine grained features. The back bone classifier uses a passthroiugh layer to furthewr bopost the granuarilty of the final feature map by concatenating a 26 by 26 lyer to the 13 by 13 final feature map. This leads to a boost in channel depth.
7. Multiscale training, images are resized in factors of 32 during training.
8. Batch normalization to stabilize training, speed up convergence and regularize the model

# Yolo v3

Key improvements

1. Added FPN for multiscale training.
2. Newer faster conv base

Results

Fastest object detector in the market
Almost as accurate as RetinaNet in mAP with IOU =0.5, but falls behind with the newer metric average mean AP.
Has trouble with larger objects now.

### Resnet

### ResneXt

The next version of Resnet. Used as the base for the second best entry for ILSVRC
2016 classification task.

Introduces another dimension called *cardinality*.

Cardinality is the division of a task into n number of tasks. Hence here n is the cardinality.

Here in ResneXT, the major difference is in each block. Each block goes through it's one by one conv as per normal RESNET. But the next step involves splitting these next 128 channels into groups of 32 (cardinality) , hence 4 input channels in each group. A 3 by 3 onv is done here and then the the result is concatenated together and then underwent a one by one conv to preserve channel size. A residual is added to maintain identity mapping.

Also *RESEARCH GROUPED CONVOLUTIONS*
This extra cardinality dimension results in better features learnt.

### MobileNet v2

Todo Explain Depthwise Seperable convs





