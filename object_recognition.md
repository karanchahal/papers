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

The step in RCNN which was to crop the blobs and run them through a feature map, was innovated upon mainly.
The innovation was a method called ROI Pooling

ROI Pooling or Region Of Interest Pooling, cropped the ROI's from the feature map instead of the input image, thus sharing computation. Also after these roi's were cropped out of a feature map and resized to a fixed size (7 by 7) , it was run though the Fast RCNN model


The Fast RCNN Network was made of a feature map, followed by a few convolutional /pooling layers. 
Then two prediction heads were attached on the final feature map. 

1. A classification head (to predict the types/class of objects) 
2. A regression/location head ( to predict coordinates of objects)

This was trained using a multitask loss which was 

Loss_of_classification + lambda(loss of regression)

lambda switched off and on if the region were background or not respectively.

Hence this sharing of computatipon resulted in 

1. Higher accuracy
2. Speed and Space efficiency (Testing on a single image reduced from 47 seconds to .32 seconds)
3. Training time reduced substantially ( from 85 hours to 9 hours)

# Drawbacks

Even though Fast RCNN was a huge improvement, still some drawbacks were consistent.

1. Bottleneck to the speed was due to the selective search procedure which took up a huge percentage of the time taken.
2. Achieved near real-time rates using very deep networks _when ignoring the time spent on region proposals_. Now, proposals were the test-time computational bottleneck.

# Faster RCNN

To solve the above drawbacks , the authors decided to iterate RCNN framework on more time. They had identified thsat the claculation of proposals took a lot of time, Hence they set out to learn how to calculate proposals.

So a major ,mileston in object recognition was achieved called the Region Proposal Network. 
This network learns to calculate effective regions of interest.

The model is as follows:
1. Apply a convolution layer on the featuire map of the image gotten after being processed by a resnet or a vgg.
2. Two prediction heads, classifiction and regression.

The RPN also brought a concept called anchors, which are predefined regions in an image , the RPN learn to calculate classification of all the anchors plus their distance from the real objects.

Anchors are defined as occuring after every x number of pixels in the image, and have different scales and aspect ratios. For example, 1:1, 1:2, 2:1 and scales 32px, 64px, 128px.
