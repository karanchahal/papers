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


# Drawbacks

The RCNN neytwork has several drawbacks 


1.  It had a complex traning procedure.Training is a multi-stage pipeline.There are 3 stages , fintuning feature map with log loss. SVM CLassiciation head to be trained to classifiy objects and infally object regression heads are trained to ouput coorindates of bounding boxes

2. Training is expensive in space and time. For SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk. With very deep networks, such as VGG16, this process takes 2.5 GPU-days for the 5k images of the VOC07 trainval set. These features require hundreds of gigabytes of storage.

3. It was slow, it took approximately 47 seconds to process a singgle image. The bottleneck as observed to be selective search which took 27 seconds on its own.

## Fast RCNN

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

# Drawbacks

Even though Fast RCNN was a huge improvement, still some drawbacks were consistent.

1. Bottleneck to the speed was due to the selective search procedure which took up a huge percentage of the time taken.
2. Achieved near real-time rates using very deep networks _when ignoring the time spent on region proposals_. Now, proposals were the test-time computational bottleneck.

# Faster RCNN

To solve the above drawbacks , the authors decided to iterate RCNN framework one more time.

They had identified that the calculation of proposals took a lot of time, Hence they set out to learn how to calculate proposals.

So a major ,milestone in object recognition was achieved called the *Region Proposal Network* . 
This network learns to calculate effective regions of interest/object proposals.

The model is as follows:
Taking an input image.

Targets are calculated taking the ground truth bounding boxes and calculating some regions of interest from that data. 

Intuitively we can think of it , as we are making the learning task a litle easier. Instead of predicting the final boxes, we tell the network to predict a large number of plausible boxes.

These target regions proposals are generated with a concept known as anchors.

What are anchors? Anchors are , simply put,  predefined crops of a 224 by 224 image.

Think of them as patches cropped after x number of pixels in the image. The size of the patches can be varied.

In Faster RCNN 9 sizes are used. These sizes are decided upon using a concept of aspect ratios of images and scales . Example: Aspect Ratios =>  1:1, 1:2, 2:1 and Scales  =>  32px, 64px and 128px. Hence 9 shapes.

### Second Major Point
These shapes are cropped after every 16 pixels in Faster RCNN

We arrive at a number 16 because the VGG Net constructs a feature map which is of 

1. Width = Image_width/16
2. Height = Image_height/16

Hence,the reason for this is we want every pixel or box in the feature map to represent one anchor position. And each anchor position has 9 values. 

* Thus the feature map with a depth dimension of 9 will represent all anchors.

* Now, we need two target values for each anchor. W want to predict what class that anchor is. Is it a dog, cat etc.

* And we also want to predict what needs to be the modification to the shape of the anchor box to better fit the object.

* For this we need offsets of the anchor height, anchor width and anchor x and anchor y coordinates.

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


### R-FCN Region Fully Convolutional Network

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

Convolutional Bases

-> Resnet
-> ResNeXt
-> Mobile Net (research efficient versions)


## Resnet

## ResneXt

The next version of Resnet. Used as the base for the second best entry for ILSVRC
2016 classification task.

Introduces another dimension called *cardinality*.

Cardinality is the division of a task into n number of tasks. Hence here n is the cardinality.

Here in ResneXT, the major difference is in each block. Each block goes through it's one by one conv as per normal RESNET. But the next step involves splitting these next 128 channels into groups of 32 (cardinality) , hence 4 input channels in each group. A 3 by 3 onv is done here and then the the result is concatenated together and then underwent a one by one conv to preserve channel size. A residual is added to maintain identity mapping.

Also *RESEARCH GROUPED CONVOLUTIONS*
This extra cardinality dimension results in better features learnt.

## MobileNet v2



