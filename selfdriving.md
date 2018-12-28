# Components of Self Driving Cars

Sensors mainly

1. Camera - dense multi spectral, no direct 3D information
2. Lidar - semi dense 3D point clouds
3. Radar - Coarse 3D signal than LIDAR too, but provides velocity also
4. GPS,IU,encoders - ego motion, know where you are

## Problems in Self Driving Cars

1. Traffic Lights
2. Pedestrains !
3. Focus information :  Remove trees and buildings
4. ....


## Decomposition Approac to Autonomy

Maps and Localisation, Perception.

Self driving cars make 3D maps to localise itself.

Use 3D Maps where you encode important informtion like (right turn only lane, lane is busy from 4-7 p.m, basically Google Maps information)

## Datasets in Perception

Kitti
- Captured in Germany
- 6 hours at 10 frames per second
- Small Dataset
- Tasks:
- 1. Primarily used for detection.
  
Cityscape 
50 cities
Several months
Good for segmentation !
- Tasks covered:
- 1. Semantic Segmentation
- 2. Instance Segmentation


Cityscapes
Toronto Dataset
Berkeley DeepDrive Dataset
Oxford RobotCar Dataset
Simulators: GTA, TORCS


# Perception

2D Object Classification and detection
3D Object Classification and detection

Start with 2D and then go onto 3D:
Common for all methods

Start by modifying a Faster RCNN mostly.

Use a single stage approach for fast results that you would need for a  self driving car setting ! 

## 3D Detection

For self driving, need to detcte and ocate lanes, traffic signs, vehicles, pedestrains, free space, cyclists etc

LIDAR has accurate depth information. Fuse 2d with Lidar to get 3D object detection. (MV3D model)

Take Lidar Point cloud and input image as input

## Tracking 

Track objects state in scene.
eg: Position, orientation, velocity, bounding box etc.


1. Deep structured model
2. End to end model

Anton Milan ETH has a good paper for tracking.

## Prediction

We need to predict what everyone else is going to do. Less amount of work being done here !

Static Object Classification
Scenario Generation- Generate possible paths, binary classification of each path liklihood independently. Normallise predictions independently.
Ballistic Prediction - Estimate path towards goal.
Things like cutting lanes, handle these tragectories.

Deep Prediction take in obejcts , map and motion plan, output predicted paths.

One option is to create a rasterised map with object polygons. Different colours for different type of objects. Add velocity etc, path of objects represent past as polygons with gradietns of brightness. 

## End to End approach  (Imitation Learning)

Hours and Hours of human driven data
process lods, extract every sensor data
Get control that humans did as training dataset

Do Data augmentation. Behaviour Cloning Approach

Very promising (does all rules, stop signs, behaves like a human)

Exponential amount of data. What to do ?

Problem is domain shift, when do,ain is shifted, policy goes crazy.