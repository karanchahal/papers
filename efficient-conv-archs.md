# Efficient Convolutaionl Architectures 


Shuffle Net is a good paper.

Have several forumlisms as to what makes an efficient CNN architecture.

1. Less MAX (memory ON access) time
2. Use same channle in and out, as it reduced MAX
3. Less group convolutions as it fragments the memory of GPU
4. Inplace operations are hecy (almost 20% more overhead)
5. 


## TODO for project

1. Include Shuffle Net in project (pretrained)
2. Include Mobile Net in project (pretrained)
3. Include DetNet in project, better backbone ?

Now with these 3 backbones
- test best speed accuracy tradeoff, it shoulkd bbe fast like shuffle net, but with more receptive field (like det net ?)
- For accuracy, port mscoco evaluation. Dont code eval yourself. 
- For speed, code average of 100 forward passes, for 1 single image. 
  

For optimsation train with adam,

1. Should you freeze backbone gradients first ?, maybe you should. Add that 



# For segmentation

1. Take trained object detection, freese backbone, train for lane markings task. 
2. Use a lightweight semantic segmentation thingy.
3. See if it trains. It should at least overfit on 1 example first. 


