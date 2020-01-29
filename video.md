# Slow Fast Networks

Main idea is to treat spatial and temporal instances in different ways unlike previous approaches. 
"We might instead “factor” the architecture to treat spatial structures and temporal events separately."
Take for example recognition,
The key idea is that the objects in the scene change slowly, for example, given a video of a person clapping their hands.
THe person does not change, and the hands clapping are still hands. So we could use a slow refresh rate for object detection. 
But an action like clapping hands: to be detected, we need a fast refresh rate for that. 

Hence, this paper introduces a two pathway neural net;

1. The first pathway captures sematic information in the video , like objects and has a slower refreshing rate. *Low* frame rates and 
*slower* refreshing speeds.
2. Second pathway represents fast changing motion, and is *fast* framerate and refreshing speed. Key idea is to make this fast pathway
lightweight.

Both these pathways are fused with lateral connections.

Note: High level relation to P cells(slow changing) and M cells(fast changing) in the human eye ball.
