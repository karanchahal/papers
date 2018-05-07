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



# Paper Draft


Introduction
how deep learning has changed the computing landscape
Deep Learning has revolutionised the computing landscape. It has led to a fundamental change in how applications are being created. Andrej Karpathy, rightfully coined it Software 2.0. Applications are fast becoming intelligent and capable of performing complex tasks. Tasks that were initially thought of being out of reach for a computer. 
Examples of these complex tasks include detecting and classifying objects in an image, summarising large amount of texts, answering questions from a passage, generating art and beating human players at complex games like Go and Chess.
The human brain processes large amounts of data of varying patterns. It identifies these patterns, reasons about them, and takes some action specific to that pattern. 
AI aims to replicate this approach through deep learning. Deep Learning has proven to have been very instrumental in understanding data of varying patterns at an accurate rate. This capability is responsible for most of the innovations in understanding language and images.
Deep Learning research is moving forward at a very fast pace , with new discoveries and algorithms coming out every few months, pushing the state of the art tasks forward.
what is object detection.
One such field that has been affected by deep learning in a substantial way is object detection. Object detection is the identification of an object in the image along with its localisation and classification. Software systems that can perform these tasks are called object detectors.



importance of object detection
Object Detection has some very important applications. A lot of tasks which require human supervision can be automated with a software system that can detect objects in images. Tasks like Surveillance, disease identification and driving can be automated to some extent with  the help of object detection. Humans put in a lot of resources to identify and detect various  phenomenon in the form of human workers. Most of these tasks can be automated by machines with the help of object detection. 
This technology can also be used very irresponsibly. Military applications come to mind where objects detection can be used to great effect. Hence, inspite of its considerable useful applications, responsibility and care should always be the first thought.
strides made in object detection, how the object detector has evolved. Faster, smaller ,accurate, more memory efficient.
Object Detectors have been making fast strides in accuracy, speed and memory footprint. Since 2015 , when the first viable deep learning based object detector came out, the field has come a long way.
The earliest deep learning object detector took 47 seconds to process and image, now it takes less than 30 ms, better than real time. Similar to speed, accuracy has also steadily improved. 
From a detection accuracy of 29 mAP (mixed average precision), modern object detectors have achieved 43 mAP.
Also, object detectors have also improved upon their size. Detectors can run well on low powered phones, thanks to the intelligent, conservative design of the models. Support for running models on phones has also helped thanks to frameworks like Tensorflow and Caffe among others. 
A decent argument can be made that object detectors have achieved close to human parity at object detection.Conversely, like any deep learning model, these detectors are still open to adversarial attacks and can misclassify objects if the image is adversarial in nature.
future work, talk about saturation in the field now.
Work is being done to make object detectors and deep learning models in general more robust to these attacks. Accuracy , speed and size will constantly be improved upon , but that is no longer the most pressing goal presently. Detectors have attained a respectable quality, that can be put into production today. The goal now would be to make these models robust against hacks and ensure that this technology is being used responsibly.

Object Detection Models

There have two approaches in constructing object detectors till now. A one step approach and a two step approach.The two step approaches have generally led in accuracy while the one step approaches have been faster and have been more memory efficient.Both these approaches have yielded substantial advancements in the field.
The one step approach classifies objects in the images in a single step.
The two step approach divides this process in two steps. The first step generates a set of regions that have a high probability of being an object. The second step then performs the final detection and classification of objects by taking these regions as input. These two steps are named the Region Proposal Step and the Object Detection Step respectively.
The one step approaches use the first step to directly predict the class probabilities and object locations.
Object detector models have gone through various changes throughout the years since 2012. The first model to come out , that really broke object detection and provided new state of the art results was the R-CNN.
There are always two components to constructing a deep learning model. The first component is responsible for dividing the training data into input and targets. The second component is deciding upon the neural network architecture.
The input for these models is an image. The targets are a list of object classes relaying what class the object belongs to and their corresponding coordinates. These coordinates signify where in the image does the object exist. The coordinates are 4 values. The centre x and y coordinates and the height and width of the bounding box. 
The network is trained to predict a list of objects with their corresponding location.
The network varies with different detector models. Popular object detection models are described below.
Two Step Models
The Region Convolutional Network (R-CNN)
The RCNN Model was the first deep learning object detection model. It is a two step object detector. Two step models have two components. The region proposal model and the object detector model. 
Region Proposal Model (RPN)
Input
The input is the image. 
A region proposal system finds a set of blobs or regions that have a high degree of possibility of being objects in that image.
The Region proposal System for  the R-CNN uses a non deep learning model called Selective Search. Selective Search finds a list of regions that it deems most plausible of having an object in them. It takes 27 seconds to run and finds a large number of regions after which the Object Detector step is run. These regions are then cropped from the input image and resized to be fed into the object detector model.
The Selective Search outputs around ~2k of regions proposals of various scales.
Object Detector Model
Input
The object detector model takes in the region proposals received from the region proposal model. These region proposals are cropped out of the image and resized to a fixed size. In the paper, the authors have used a shape of 7 by 7. These resized regions are fed into the object detector model of the RCNN.
Targets
The targets for the RCNN network are a list of class probabilities for each region proposed by the Selective Search. The box coordinate offsets are also calculated , so that the network can learn to better fit the object. Offsets modify the original region’s shape.

The region is assigned a class by calculating the IOU between it and the ground truth. If the IOU > 0.7, it is assigned the class. If multiple ground truth’s have an IOU > 0.7 with the region, the ground truth with the highest IOU is assigned to that region. If the IOU < 0.3, it is assigned the background class, other regions are not used in calculation of the loss, hence ignored in training.

If an anchor is allotted a foreground class, offsets between the ground truth and the anchor are calculated. These box offsets targets are calculated to make the network learn how to better fit the ground truth by modifying the shape of the anchor. 

The method for calculating these offsets varies. In the RCNN series, it is calculated as follows:

tgx = (x_g - x_a)/w_a
tgy = (y_g - y_a)/h_a
tgw = log(w_g/w_a)
tgh = log(h_g/h_a)

Architecture
The architecture of the Object Detector Model consists of a set of convolutional and max pooling layers with activation functions . A region from the above step is run through these layers to generate a feature map. This feature map denotes a high level format of the region that is interpretable by the model. The feature map is unrolled and fed into two fully connected layers to generate a 4078 dimensional vector. This vector is then input into two separate small SVM networks. The classification network head and the regression network head.
The classification head is responsible for predicting the class of object that the region belongs to.The regression head is responsible for predicting the offsets to the coordinates of the region to better fit the object.
Loss
There are two losses computed in the RCNN, classification loss and the regression loss. The classification loss is the cross entropy loss and the regression loss is a L2 loss.
The RCNN uses a multi step training pipeline, to train the network using two losses.
Model Training Procedure
The model is trained using a two step procedure.Before training, a pre trained convolutional base from ImageNet is used.
The first step includes training the SVM classification head, using the cross entropy loss. The weights for the regression head are not updated.
In the second step, the regression head is trained with the L2 loss. The weights for the classification head are fixed.
This process take a long time, approximately 84 hours. This huge time is due to computing features for each region proposal and storing them. The high number of regions take up a lot of space and the IO operations add a substantial overhead.
Salient Features 
The RCNN model takes 47 seconds to process an image
It has a complex multistep training procedure, that involves careful tweaking of parameters.
Training is expensive for both time and space. The features computed for the dataset take hundred of gigabytes in space and 84 hours are required to train the entire model.
A more efficient model was needed, but RCNN had provided a good base to iterate upon. It had provided a structure to solve the object detection problem, by breaking the problem into 2 steps. Now these ideas could be iterated upon.

Fast RCNN
The Faster RCNN came out soon after the RCNN and was a substantial improvement upon the original model even though the structure remained the same. The Fast RCNN is very similar to the RCNN, in that it uses selective search to find some regions and then runs each region through the object detector network. This network consists of a convolution base and two SVM heads for classification and regression. Predictions are made for the class of that region and offsets to the region’s coordinates in the base image. 
The RCNN took every region proposal and ran them through the convolutional base. This was quite inefficient as an overhead of running a region through the convolutional base was added for each region proposal. The Fast RCNN aimed to reduce this overhead by running the convolutional base just once. Fast RCNN ran the convolutional base over the entire image to generate a feature map. The regions are cropped from this feature map instead of the input image.This cropping procedure is done using a new algorithm called ROI Pooling.
Using ROI Pooling, features are shared leading to a reduction in both memory space and time.
The second change in the Fast RCNN introduced a single step training pipeline and multitask loss to make training the classification and regression heads simultaneously possible.These changes led to substantial decrease in training time and memory needed. 
Fast RCNN is more of a speed improvement than a accuracy improvement. It takes the ideas of RCNN and packages it in a more compact architecture.
The improvements in the model included a single step end to end training pipeline instead of a multi step one and reduced training time from 84 hours to 9 hours. It also had a reduced memory footprint, no longer requiring features to be stored on disk.
The major innovation of Fast RCNN to achieve this was the sharing of features of a convolutional net and constructing a single step training pipeline which trained a multitask loss instead of a two step training pipeline.

ROI Pooling

Region of Interest Pooling or ROI Pooling is a algorithm that takes the coordinates of the regions obtained via the Selective Search step and directly crops it out from the feature map of the original image. 

ROI Pooling allows for computation to be shared for all regions, as the convolutional base need not be run for each region. The convolutional base is run once , for the input image to generate a single feature map. Features for various regions are computed by cropping on this feature map. 

The ROI Pooling algorithm works as follows.

The coordinates of the regions proposed are divided by a factor of h, the compression factor. The compression factor is the amount by which the image is compressed after it is run through the convolutional base. It is usually 16, as the VGGNet compresses the image to be 1/16th its size in width and height.

Once the compressed coordinates are calculated, they are plotted on the image feature map. The region is cropped and resized from the feature map to a size of 7 by 7. 

This resizing is done using various methods in practice. In ROI Pooling, the region plotted on the feature map is divided into 7 by 7 bins. Max pooling is done on the cells in each bin. ROI Averaging is a variation on ROI Pooling and it takes the average instead of max pooling. The procedure to divide this feature map into 49 bins is approximate. It is not uncommon for one bin to contain more number of cells than the other bins.

Modern object detectors simply resize the cropped region from the feature map to a size of 7 by 7 using common image algorithms instead going through the ROI Pooling step. In practice, accuracy isn't affected substantially.

Once these regions are cropped, they are ready to be input into the object detector model.

Object Detector Model
The object detector is the same detector used for the RCNN. The only change is that the input comes from the ROI Pooling step.
Input
The object detector model takes in the region proposals received from the region proposal model. These region proposals are cropped using ROI Pooling.
Targets

Targets are computed for these region proposals in a similar way as the RCNN. Class probabilities and box regression offsets are calculated.
Architecture
The architecture of the Fast RCNN object detector model is very similar to the RCNN. After going through the ROI Pooling step, the cropped regions are run through the convolutional layers and into the two classification and regression SVM heads. The convolutional layers are a set of convolutional and fully connected layers. These layers output a 4078 dimensional vector. This vector is input into the classification and regression heads , similar to the RCNN.

Loss
A multitask loss used is as follows:

loss_of_classification + alpha*lambda*(loss of regression)]

Alpha is switched to 1 if the region was classified with a foreground class and 0 if the region was classified as background. The intuition is that the loss generated via the regression head should only be taken into account if the region actually has an object in it.

The loss of classification is a regular log loss and the loss of regression is a smooth L1 loss. The Smooth L1 loss is an improvement over the L2 loss used for regression in RCNN. It is found to be less sensitive to outliers as training with unbounded regression targets led to gradient explosion. Hence ,a very careful tuned learning rate needed to be followed. Using the L1 loss removed this problem.

Lambda is a weighting factor which controls the weight given to each of these losses. It is a hyper parameter and is set to 1 in training the network.

This loss enables joint training.

Model Training Procedure
In the RCNN, training had two distinct steps.The Fast RCNN introduced a single step training pipeline where the classification and regression subnetworks could be trained together using the multi task loss described above.

The network is trained with SGD with a mini batch size of 2 images. 64 random ROI’s were taken from each image resulting in a mini batch size of 128 ROI’s.

Salient Features

It shared computation through the ROI Pooling Step hence leading to dramatic increases in speed and memory efficiency. More specifically, it reduced time to train exponentially (84 hrs to 9 hrs) and also reduced inference time from 47 seconds to .32 seconds.

Introduced a simpler single step training pipeline and a new loss function. This loss function was easier to train and did not suffer with gradient explosion problems.

The region proposal step was the bottleneck now, the Fast RCNN reported real time speeds, without the Selective Search.

Faster RCNN

Faster RCNN came out soon after the Fast RCNN paper. It was meant to represent the final stage of what the RCNN set out to do. It proposed a detector that was learnt end to end. This entailed doing away with the algorithmic region proposal selection method and constructing a network that learnt to predict good region proposals. Selective Search was serviceable but took a lot of time and set a bottleneck for accuracy. A network that learnt to predict higher quality regions would theoretically have higher quality predictions .

The Faster RCNN came out with a Region Proposal Network (RPN). The RPN learnt to predict good proposals and replaced the Selective Search. 

The RPN network needed to have the capability of reciting regions of multiple scales and aspect ratios. This was achieved using a novel concept of anchors. 




Anchors

Anchors are a set of regions of predefined shape and size in an image i.e anchors are simply rectangular crops of the image. The sizes of anchors are of different sizes and aspect ratios. The multiple shapes are decided by coming up with a set of aspect ratios and scales.

The authors use areas/scales of {32px, 64px, 128px} and aspect ratios of {1:1,1:2,2:1}. Hence, there are 9 types of anchors . The next step is finding out the pixels in the image out of which these 9 anchors are to be cropped. 	

Anchors are not cropped out of every pixel, but after every x number of pixels in the image. This process starts at the top left and continues on until we reach the bottom right of the image. This scanning is done from left to right ,x pixels at a time, moving x pixels towards the bottom after each horizontal scan is done. This is also called a sliding window.

The number of pixels after which a new set of anchors would be cropped out are decided by the compression factor of the feature map described above. In VGGNet , that number is 16. Hence in the paper, 9 crops of different sizes are cropped out after every 16 pixels (height and width) in a sliding window fashion across the image.

Anchors cover each and every region of the image quite well. Faster RCNN uses the concept of anchors to cover all possible regions in an image.

Region Proposal Model

Input

The input to the model is the input image. The input training data for the model is augmented by using standard tricks like horizontal flipping. Images are of a fixed size, 224 by 224. To decrease training time, batches of images are fed into the network. The GPU can parallelise matrix multiplications therefore it can process multiple images at a time.

Targets

Before going into the details of the architecture of the model, a description of the targets that the network trains against needs to be described.

Once the entire set of anchors are cropped out the image, two parameters are calculated for each anchor. The class probability target and the box coordinate offset target.

The class probability target for each anchor is calculated by taking the IOU of the ground truth with the anchor. If the IOU is greater than 0.7, we assign the anchor the class of the ground truth object. If there are multiple ground truths with an IOU greater than 0.7, we take the highest one. If the IOU is less than 0.3, we assign it the background class. If the IOU is between 0.3 and 0.7, we fill in 0 values, as those anchors won’t be considered when the loss will be calculated, hence not affecting training.

If an anchor is allotted a foreground class, offsets between the ground truth and the anchor are calculated. These box offsets targets are calculated to make the network learn how to better fit the ground truth by modifying the shape of the anchor. 

The method for calculating these offsets varies. In RCNN series, it is calculated as follows:

tgx = (x_g - x_a)/w_a
tgy = (y_g - y_a)/h_a
tgw = log(w_g/w_a)
tgh = log(h_g/h_a)

Architecture

The RPN network consists of a convolutional base similar to the one used in the RCNN object detector model. The convolutional base outputs a feature map. This feature map is fed into two subnetworks. A classification subnetwork and a regression subnetwork.


The classification and regression head consist of few convolutional layers which generate a classification and regression feature map respectively.The only difference in the architecture of the two networks is the shape of the final feature map.

The cells in the feature map denote the set of pixels out of which the anchors are cropped out of. Each cell has a depth which represents the class probabilities of each anchor type.

The classification feature map has dimensions w x h x (k*m). Where w and h are the width and height. k denotes the number of anchors per pixel point, which in this case is 9. m represents the class probabilities  including the background class.

The cells in the feature map denote the set of pixels out of which the anchors are cropped out of. Each cell has a depth which represents the box regression offsets of each anchor type.

Similarly like the classification head , the regression head has a few convolutional layers which generate a regression feature map. This feature map has dimensions w x h x (k*4). The 4 factor is meant to represent the predictions four offset coordinates for each anchor.

Loss

These anchors are meant to denote the good region proposals that the object detector model should further classify on.

The model is trained with simple log loss for the classification head and a new smooth L1 loss for the regression head. There is a weighting factor of lambda that balances the weight of the loss generated by both the heads, that is similar to the Fast RCNN loss.

Smooth Loss equation here

There loss can be computed across all anchors but the model doesn’t converge. The reason for this is that the training set is dominated by negative/ background examples. To evade this problem, the training set is made by collecting 256 anchors to train on. 128 of these are foreground anchors and 128 are background anchors. Challenges have been encountered keeping this training set balanced. It is an active area of research.

Object detector model


The object detector model is the same Fast RCNN object detector model. The only difference is that the input to the model comes from the proposals generated by the RPN and not the Selective Search.


Loss

There are 4 losses that need to be combined to enable end to end training.

The 4 losses are the classification and regression losses for the RPN , and the classification and regression losses for the Fast RCNN.

These losses are combined using a weighted sum.

Faster RCNN Model Training Procedure
The whole network is trained using a joint approach.
SGD is used and the model converges in the same time as the Fast RCNN. Images were resized to various shapes to facilitate multi scale training.


Salient Features

Faster RCNN is faster and has a full neural network pipeline.

It runs at near real time.

The network is more accurate as the RPN improves region proposal quality and hence the detector’s overall accuracy.

Extensions to Faster RCNN


There have been various extensions to Faster RCNN network to make it faster and more scale invariant. 

To make the Faster RCNN scale invariant, the original paper took the  input image and resized it to various sizes. These images were then run through the network. This approach wasn’t ideal as the network ran through one image multiple times making the object detector less than real time. Feature Pyramid Networks provided a robust way to deal with images of different scales.

Feature Pyramid Networks (FPN)

FPN is a neural network that trains a network to be invariant to scale. 

In the original Faster RCNN, a single feature map was created. A classification head and a regression head was attached to this feature map. In an FPN however, there are multiple feature maps, that are designed to represent the image at different scales.

The regression and classification heads are then run across all of these multi scale feature maps. A good side effect of this is anchors no longer have to take care of scale. They can only represent aspect ratio as the scale is handled by these multi scale feature maps implicitly. 

Architecture

The FPN model takes in an input image, and runs them through the convolutional base. A modern convolutional base takes the input through various scales, steadily transforming the image to be smaller in height and width but deeper in channel depth. In a ResNet , the image goes through 5 scales; 224, 56, 28, 14, 7 respectively.

The model works has follows, feature maps of each scale are taken. These feature maps are the last layer of that scale, as going deeper is always better.

Each feature map goes through a 1 by 1 convolution to bring the channel depth to be uniform. The authors used 256 channels in their implementation. 

Now, these feature maps are element wise added with the upsampled feature map of the scale directly above that specific feature map. These are called lateral connections. The upsampling is done using nearest neighbour sampling with a factor of 2 . This upsampling procedure across various feature scales are called top down pathways. he processes for making an image smaller steadily are called bottom up pathways.

Once lateral connections have been performed for each scale, those feature maps go through a 3 by 3 convolution to generate the final feature maps for each scale.

This procedure of lateral connections that merges bottom up pathways and top down pathways ensures that the feature maps have a high level of information while still retaining it localisation information of each pixel. It forms a nice compromise between getting more salient features but still retaining the overall structure of the image at that scale. 

After generating these multiple feature maps, the Faster RCNN works are normal. The only difference being now, the regression and classification heads run on multiple feature maps instead of one. 

FPN’s allowed for invariance in testing and training imagers. Previously, Faster RCNN was trained on multi scaled images but testing on images of a single scale.

Now due to the structure of FPN, multi scale testing was done implicitly.

Salient Points

New state of the art were set in object detection, object segmentation and classification by integrating FPNs in the pre-existing models.

4 feature maps are constructed in the author’s version of FPN. They signify anchors of scales 32px, 64px, 128px and 256px.

ROI Pooling is done on the feature map, to which the anchor belongs.

Multi-scale testing was possible in real time with FPN’s. 

There is one more bottleneck in the Faster RCNN architecture, when the proposals are predicted from the RPN. Each proposal has to be cropped out from the feature map, resized and fed into the Fast RCNN network.

Hence the model isn’t fully convolutional. Another paper seeks to optimise this part and hence increase the speed of the model. 

Region - Fully Convolutional Network (R-FCN)

This network refactors the Faster RCNN network such that it is fully convolutional. A fully convolution network consists only of convolutional layers with no fully connected layers.

There are several benefits of a fully convolutional network. One of them is speed, computing convolutions is faster than computing a fully connected layer. The other benefit is that the network becomes scale invariant, images of various sizes can be input into the network without modifying the architecture because of the absence of a fully connected layer. Fully convolutional networks first gained popularity with segmentation networks.

Architecture

The R-FCN model modifies the object detection model in the Faster RCNN. Instead of cropping each ROI out of the feature map. The model  diverges into two heads and modifies the feature map by undergoing a 1 by 1 convolution for each to bring the depth to a size of z_c and z_r for the classification and the regression head respectively.

The classification head’s z_c value is k*k*(x), where k is 7 , which was the size of the side of the crop after ROI Pooling. x represents the number of classes, including the background class. The value of is z_r is k*k*4. The 4 representing the 4 box coordinates.

The predictions for a region proposal is calculated as follows.

Since, each class has k*k channels on the feature map. 
The process of cropping a region is very similar to ROI Pooling. But instead of max pooling the values from each bin on the single feature map. Max pooling is done on different feature maps. These feature maps are chosen depending on the position of the bin. For example if max pooling is needed to be done for the i th bin out of k*k bins, The i th feature map would be used for each class. The coordinates for that bin would be mapped on the feature map and a single max pooled value is calculated.

Explain why this method is better.

Using this method a ROI map is created out of the feature map. The probability value for a ROI is then calculated by simply finding out the average/max of this ROI map.

Vectors are created for the classification head and the regression head. The vectors are of size x and 4 respectively. x being the number of classes. A softmax is computed on the classification head to give the final class probabilities. 

Hence, a R-FCN modifies the ROI Pooling and does it at the end of the convolutional operations. There is no extra convolution layers that a region goes through, after the ROI Pooling. R-FCN’s share features in a more effective way than a Faster RCNN. Speed improvements are reported while also conserving accuracy.

Salient Features

The R-FCN sets new state of the art in speed of two step detectors , achieving a test time speed of 170ms, which was 2.5 to 20x stranger than it’s Faster RCNN counterpart.

The intuition behind it was that this method mangoes to address the dilemma between invariance/variance on translation. Hence fully convolutional classification nets like ResNets can be effectively converted to fully convolutional object detectors.


Single Step Object Detectors


Single Step Object Detectors have been popular for some time now. Their simplicity and speed with reasonable accuracy have been powerful reasons for their popularity.

Single step detectors are constructed as a modification of Fast RCNN, they do not have a region proposal step, and directly predict object classes and coordinate offsets.

Single Shot MultiBox Detector (SSD)

This network came out in paper, which boasted state of the art results in detection at the time, while being faster than all the alternatives. The SSD uses a very similar concept as YOLO. In that, it uses anchors to define number of default regions in an image and these anchors predict t the class scores and the box coordinates offsets. A backbone convolutional base (VGG16) was used and a multitask loss was computed to train the network. This loss was very similar to the Yolo/ Faster RCNN loss. It had a smooth L1 loss to predict the box offsets and a cross entropy loss to train for the class probabilities. The major difference in SSD from the YOLO and Faster RCNN architectures was that it was the first model to propose training on a feature pyramid. Leveraging the structure of convolutional layers to go smaller per layer, to form a multi scale pyramid. 

The network was training on n number of feature maps, instead of just one. These feature maps, taken from each layer, very similar to the FPN network. But unlike the FPN network, it does not use top down pathways to enrich the feature map with higher level information . A feature map is taken from each scale and a loss was computed and back propagated. 

The SSD network computed the anchors and allowed them to each scale in a unique way. The network, used a concept of aspect ratios and scales. Each point on the network, generated 6 anchors. These anchors varied in aspect ratio. The same point on each feature map, varied in scale. SSD uses this feature pyramid to achieve a very good accuracy, while remaining the fastest detector on the market.


You only look Once Detector (YOLO)

The YOLO architecture was constructed in the same vein as the Fast RCNN. The image was run through a few convolutional layers to construct a feature map. The concept of anchors was used here too, with every grid cell acting as a pixel point on the original image. The YOLO algorithm generated 2 anchors for each grid cell.

Unlike the Fast RCNN, Yolo has only one head. The head outputs feature map of size 7 by 7 by (x+1+ 5*(k)).

K = number of anchors
x+1 = number of classes plus background class
5 = 4 offsets of x, y, height and width. 1 extra parameter to detect if region contains object or not. YOLO coins it as the “objectness” of the anchor.


A multi task loss is used to train the network end to end. 

Offset Calculation

Yolo uses a different formulation to calculate offsets. Offsets calculation  in Faster RCNN is done as follows:

tgx = (x_g - x_a)/w_a
tgy = (y_g - y_a)/h_a
tgw = log(w_g/w_a)
tgh = log(h_g/h_a)

Here tgx are the targets

Predictions are as follows:

tx = (x - x_a)/w_a
ty = (y - y_a)/h_a
tw = log(w/w_a)
th = log(h/h_a)

tx and ty are the predictions.

This formulation worked well for faster RCNN , but the authors of YOLO point out , that this formulation is unconstrained, and offsets can be predicted such that the anchor can be modified to be anywhere in the image. In practice , it takes a long while for the model to start predicting sensible offsets.
They hypothesised that this was not needed as an anchor at a pixel point would only be responsible for modifying its structure around the anchor box.

YOLO came up with a new formulation for offsets, that constrained the predictions of these offsets to near the anchor box.  
The new formulation modified the above objective by training the network to predict these 5 values.

b_x = sigmoid(tx) + c_x
b_y = sigmoid(ty) + c_y
b_w = w_a*e^tw
b_h = h_a*e^th
b_o = sigmoid( prediction for objectness )

The new formulation better constrains the prediction to around the anchor box.


RetinaNet

The people at FAIR who are the pioneers of modern object detectors came out with a new detector which builds further upon the architecture by introducing a novel loss function. This model is a 1 step detector but boosts state of the art accuracy. 

The authors realised that the reason why one step detectors had lagged behind 2 step detectors in accuracy was the implicit class imbalance problem while training. 

Class imbalance occurs when the number of different training examples are not equal. It happens in single step detectors due to extreme foreground/background imbalance.

2 step detectors evade this problem, by using RPN’s to bring down the number of anchors to a manageable number of 1-2k . Then, for the next step, techniques like a fixed foreground-background ratio of 1:3 or online hard example mining are performed to maintain a manageable balance between foreground and background examples. 

But conversely, single step detectors densely samples regions from all over the image, leading to a number of around 100k anchors. Similar techniques like hard example mining and fixed foreground/background ratios can be implemented but they are inefficient as the procedure is still dominated by easily classified background examples.

Retina Net proposes a dynamic loss function which down weights the loss contributed by easily classified examples. The scaling factor decays to zero when the confidence in predicting a certain class increases. This loss function can automatically down weight the contribution of easy examples during training and rapidly focus the model on hard examples. 

The Object Detector Model

RetinaNet uses a simple object detector , deriving from all the best practices found by the above research. 

It uses anchors to predict regions and uses a FPN network for training for various scales. It uses a single unified network composed of a backbone network and two task specific subnetworks.The first subnetwork predicts the class of the region and the second subnetwork predicts the coordinate offsets. 

In the paper, the authors have used a Resnet, augmented by a FPN to create a rich feature pyramid. 

The detector uses anchors to predict regions. Each level on the feature pyramid uses 9 anchor sizes at each location. The original set of 2 aspect ratios {1:1, 1:2, 2:1 }, have been augmented by the factors of {1, 2 ^1/3 , 2^ 2/3} for denser scale coverage.

Hence , the anchors cover a scale range of 32-813 pixels with respect to input image.

Each anchor is assigned a set of K object classes (including the background class) and 4 bounding boxes similar to Yolo. The only difference is that there is no “objectness” feature. 

Targets

Targets for the network are calculated as follows. Class Probability Targets are set as 1 for a object class , if the IOU between the ground truth box and the anchor is more than 0.5. It is given a background class if the IOU is less than 0.4 . If the IOU lies between 0.4 and 0.5, that anchor is ignored in training. 

Box regression targets are computed as the offset between each anchor and its assigned ground truth box, no targets are calculated if the anchor belongs to the background class.

Architecture

The classification and the regression subnets are quite similar in structure. Each pyramid level is attached with these subnetworks, which share weighs across all levels. A small FCN is attached, consisting of 4 convolutional layers of filter size 3 by 3. Each convolutional layer has a RELU attached to it and maintains the same channel size as the input feature map. Finally sigmoid activations are attached to output a feature map of depth A*K. Where A = 9, representing the number of aspect ratios per anchor and K represents the number of object classes.

The box regression subnet is identical to the classification subnet excerpt fro the last layer. The last layer has the depth of 4A. 4 indicating the width, height and x and y coordinate offsets. The regime for calculating the offsets is the same as in Faster RCNN. 

The authors claim that a class agnostic box regressor is equally accurate while also having fewer parameters.

Loss

Focal loss is used to train the entire network. This is the major innovation of RetinaNet and what allows it to be much more accurate than its counterparts.

The network architecture is constructed in a. way that there is a huge imbalance in the number of background and foreground training examples. Methods to combat class imbalance have been used in the past , the most common being the balanced Cross entropy loss.

The balance cross entry loss down weights the effect of loss generated by the background class hence reducing it’s effect. This is done using a hyper parameter called alpha. 

balanced_cross_entropy = alpha * cross_entropy. 
The value for alpha is a for class 1 (foreground) and 1- alpha for class -1 (background).
The value for alpha could be the inverse class frequency or treated as a hyperparameter to be set by cross validation.

This loss does balance the importance of positive/negative examples , but it does not differentiate between easy/hard examples. 
The focal loss proposes reshaping the loss function so that it down weights easy examples and thus focuses on training hard negatives.

This is achieved through a modulating factor of (1-loss)^y. 

Where y is a scaling factor, generally set to 2.

Hence the total focal loss is parameterised by:

FL = (1- loss)^y * cross_entropy 

A alpha weighting factor is added to balance between foreground and background. 

Hence 

balanced_FL = alpha * (1- loss)^y * cross_entropy


The modulating factor down weights the effect of the loss if the examples are easy to predict. The factor of y adds an exponential factor to the scale. 

For example, the loss generated by a example, predicted to be 0.9, will be down weighted by a factor of 100x while a example predicted to be 0.99, will be down weighted by a factor of 1000x.


Training and Inference

Training is done using Stochastic Gradient Descent (SGD), with an initial learning rate of 0.01 , which is divided by 10 after 60k examples and and again after 80k examples. SGD is initialised with a weight decay of 0.0001 and a momentum of 0.9. Horizontal image flipping is the only data augmentation technique used. 

While training at the start, the focal loss fails to converge and diverges early in training. To combat this, the network always predicts a probability of 0.01 for the foreground class, to stabilise training.

Inference is done by running the image through the network. Only the top 1k predictions are taken at each feature level after thresholding detector confidence at 0.05. NMS is performed with a threshold of 0.5 and the boxes are overlaid on the image to form the final output. This technique is seen to improve training stability for both cross entropy and focal loss in the case of heavy class imbalance.









Metrics To Evaluate Object Recognition Models

There have been several metrics that the research community uses to evaluate object detection models. These are listed here.


Mixed Average Precision

Mixed Average Precision describe more


Convolutional Bases

Modern Object detectors have a convolutional base which computes a feature map. This feature map depicts a high level description of the image. Through a series of convolutions that make the image smaller and deeper, the network aims to make sense of the various shapes in the image. Convolutional Networks form the backbone of most modern computer vision models. 

There have been a lot of convolutions networks that have come out these past few years. A few examples are AlexNet, VGGNet, Resnet.

Nowadays , convolutional networks are segregated on three factors.

Accuracy, Speed and Memory. Modern object detectors use the convolutional bases that are most apt for their use case. For example, to put an object detector on the phone, the network should be small and fast. Alternatively, where accuracy is the most important thing, big nets are used, generally hosted on the cloud, with powerful GPU’s.

A lot of research has gone into making these convolutional nets faster and more accurate. A few popular detectors are described here.

Resnet

This is an extremely popular convolutional network, made by stacking convolutional layers with a concept called skip connections. Skip connections simply add / concatenate the features of a previous layer to the current layer. It is shown that , by doing this the network propagates gradients much more effectively during backpropogation. Resnets were the state of the art at the time they were released and are still quite popular today. The innovation of introducing skip connections resulted in the capability of training very deep networks without overfitting.

Resnets are usually used with powerful GPU’s as their processing would take substantially more time to do on a CPU. These networks are a good choice for a backbone on a powerful cloud server.


Resnext 

Resnext networks are a further iteration on the resent by adding grouped convolutions as another dimension to the convolution. Convolutions operate in three dimensions namely, width, height and depth. Resnext brings a new dimension called cardinality. Cardinality espouses dividing a task into n number of smaller subtasks. Cardinality represents the number n. 

Each block of the network goes through a 1 by 1 convolution similar to the Resnet to reduce dimensionality. The next step is slightly different. Instead of running the map through a 3 by 3 convolutional layer, the network splits the m channels into groups of  n, where n is the cardinality.

A 3 by 3 convolution is done for each of these n groups and the groups are then concatenated together. After concatenation this aggregation undergoes another 1 by 1 convolution layer to adjust the channel size. Similar to resnet, a skip connection is added to this result.

Resnext networks use of these grouped convolutions lead to a better classification accuracy , while still maintaining the speed of a Resnet network. 

MobileNets

These networks are a new line of convolutional networks made with speed and memory efficiency in mind. Mobile Nets are used on Lowe power devices like smart phone, embedded systems etc. 

Mobile Nets introduced depth wise separable convolutions which led to a major loss in floating point operations, while still retaining comparative accuracy. 

Depthwise separable convolutions modified the convolutional procedure . The traditional procedure of converting a 16 channels to 32 channels would be to do it in one go. The floating point operations for this are w*h*3*3*16*32 for a 3 by 3 conv.

The modified procedure is first to make the feature map go through a three by three convolution by not merging anything, hence we have 16 feature maps. Then we take these 16 feature map and apply a 1 by 1 filter with 32 channels resulting in 32 feature maps.
Hence the total computation is (3*3*16 + 16*32*1*1) , which is far less than the above.

Depthwise Seperable Convolutions from the backbone of Mobile Nets. 

Since then some other small, accurate models constructed for mobile in mind have come out like Effnet, Shufflenet that are quite promising.
Speed vs Accuracy Comparisons









