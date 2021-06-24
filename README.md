# Feature Aligment

Code for the paper: *Feature Alignment for Approximated Reversibility in Neural Networks*
https://arxiv.org/abs/2106.12562

Each folder has the same files, but shaped for different data sets.
The file *training.py* contains the main code which should be executed. 
The file *networks.py* contais the networks architetures for the main file.
The file *memory.py* trains a neural network locally as discussed in the paper.

If you wish to visualize the reconstructed images or sample new images, you can
run the *sample.py* file for sampling non-local feature aligment, *interpolation.py*
to interpolate between four pairs of images and *memory_rec.py* to reconstruct
images locally.

--------------
The code was tested on the framework PyTorch version 1.9.

Please let me know if you see any mistake.

MIT LICENCE