# Facts about training

All training is done on 20 epochs.

__MODEL ARCH__ : __RESNET18__

All training is done with an SGD optimiser with momentum

__optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)__

20 epochs take  __28 minutes__ on Collab GPU.


1. We train for 20 epochs with 1 cycle policy. Learning rate 0.001 to 0.01 and back. 
The network overfits at __94%__. Get a test accuracy of __90.67__. 
_Accuracy keeps increasing till 20 epochs_. __NO ANNIHILATION__


2. We train for 20 epochs with 1 cycle policy with  Learning rate 0.01 to 0.1 and back. network overfit s at **96%**. test accuarcy is **91.31**.
_Accuracy keeps increasing till 20 epochs_.  __NO ANNIHILATION__

3. We train for 20 epochs with 1 cycle policy with  Learning rate 0.01 to 0.8 and back TILL 16 Epochs. Then __4 EPOCHS__ of Learning rate annihilation . That means LR dropped to 0.001 and then trained.
Network overfit at __94%__. Test accuracy is __90.6__ 
_Accuracy converges till 20 epochs_.  __WITH ANNIHILATION__

4. We train for __24__ epochs with 1 cycle policy with  Learning rate 0.01 to 0.1 and back. Then we train network for 2 epochs at __0.001 LR__ .network overfit s at **98%**. test accuarcy is **93.4**.
_Accuracy converges till 26 epochs_.  __WITH ANNIHILATION__

5. We train for __28__ epochs with 1 cycle policy with  Learning rate 0.01 to 0.1 and back. Then we train network for 4 epochs at __0.001 LR__ .network overfit s at **98%**. test accuarcy is **93.2**.
_Accuracy converges till 32 epochs_.  __WITH ANNIHILATION 4 EPOCHS__



**Epochs is a better measure than time !!**


# Thoughts

1. Haven't tried using AdamW
2. Heven't tried decreasing momemtum with increasing LR and vice versa.
3. Haven't checked batch sizes weith claims.
4. Used only resnet18.
