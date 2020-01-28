# Self supervised Learning

1. Jigsaw: Patches are shuffled, model predicts order of these patches, which are shuffled randomly. 
2. Tranferable to video. 
3. For language: Example is BERT.

4. Difference between CV and NLP:
- NLP , the signal is discrete.
- CV, signal is high dimensional, continous. 

### Recent Successes

MoCo - Momentum Contrast for SSL. Train end to end, not finetune.

PIRL - Semantic Representations should be invariant under image transformations. 
Task - Jigsaw. 

Contrastive Predictive Coding- Precit encodings below a certain image patch. 

Sparse Overcomplete Representation of Image Data with a Multi Layer Conv Layer-

Idea: Use self supervision to learning sparse overcomplete representation of image.
1. Put in image, get back same image. 


Why sparsity ?
An inductive bias only . small number of factors are resposible for a single data point. 
Why overcompleteness. 
Enables model flexibility. 
More robust to noise.

- Sparse Coding and Dictionary Learning. 
