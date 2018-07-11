# Cutting Edge Deep Learning Lesson 8 Notes

This cutting edge deep learning class has the following features:

1. Teaches approaches to solve __unsolved__ problems
2. Operates on the __cutting edge__ of research.
3. Some techniques are __extremely new__.

## Object Detection


The lecture tackles single object detection in the image first.

This process is divided into a few steps

## Step 1: Classify and Localise the largest bounding box in the Image

### Loading the Pascal VOC Dataset

Download Pascal 2007 dataset from the website. Then to load the dataset we exeute the follwoing code 

```python
PATH = PATH('<path-to-pascal-data>') # PATH is a handy library to confiure paths in filesystem
list(PATH.iterdir(()) # gets all filenames in the directory

#open a file

trn_j = json.load(PATH/'one.json').open())
trn_j.keys()
```

__Hint: You can name some data types as CONSTANTS for auto completion as given in the below example__
```python
FILE_NAME, IMG_ID, CAT_ID, BBOX = 'file_name','id','image_id','category_id','bbox'
```
Use defaultdict in python if you want to create a default dictionary for new keys.

```python
trn_anno = collections.defaultdict(lambda:[])
# load dict with key being image_id and value being bounding box and category id of the object.
# fast ai is going to refer rows by columns always.
# bbox coordinates consist of left top x,y and bottom right x,y
```


__Hint: Create a function to transalte bbox coordinates back into x,y,height and width format__


### Open CV

Fast.ai uses opencv because of 
1. OpenCV is 5-10x faster than Pillow, Scikit-Image
2. Pillow is not very thread safe
3. Because of GIL (Global Interpreter Lock), python imposes a big lock which prevents two threads from doing pythonic things. This is a disadvantage of python.
4. Open CV, doesn't use Python to launch threads. Open CV runs thread in C. Hence very fast.

Disadvantages
1. OpenCV has a very shitty API.

So use Open CV , as it __fast__.

### Matplotlib

Matplotlib was very shitty before, the folks there decided to maintain a new very useful object orientated wrapper. No examples 
online to understand the much better API.

#### Tricks

1. plt.subplots
```python
fig,ax = plt.subplots(figsize=figsize)

ax.imshow(im) #displays image
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

```
It returns figure and axis.The use axis to set various thing on the plot

We can draw outlines on boxes via matplotlib.

## Predict Class

### Steps

Load dataset. That means load images and its corresponding bounding box. First we look at only generating the bounding box for thr largest object in an image.


We filter the annotations so that we only take the largest bounding box for the image.

Now, that we have our dataset. We load the dataset into fast.ai by saving our data to a csv and then using the __inbuilt fastai dataloader__ 
to process our data.

The easiest way to save to the csv is __pandas DataFrame__.

Then we train a simple resnet classifier to predict the class of largest object.

#### PDG Python Debugger

--- ```pdb.set_trace()```
--- ```%debug``` pops open the debugger at the last exception

## Predict Bounding Box Coordinates

To predict bounding box around the object, we construct a regresiion model to predict the 4 coordinates. The MSE loss is good for 
regression, so we'll use that.  We'll use a variation of MSE called L1 loss which is much better for unbounded inputs. MSE penalises too much,
hence L1 is better.

We lr_find to find the learning rate.

So in short, we create a single object detector classifier. Trained with the latest techbiques using a simpel classification and regression head
Both are trained seperately.





