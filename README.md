# Machine-Perception-Project1-
Description: Implement one attack model (FGSM or PGD) on the Cifar-10 data. Compare the drops in classification performance for three values of noise magnitude.
Members: Johnny Lin, Matt Wei, Boshen Pan

This project trains and evaluates models on the CIFAR-10 dataset, employing a Deep Convolutional Neural Network (DCNN) and a Vision Transformer (ViT) architecture. Additionally, it demonstrates the impact of adversarial attacks on model performance using FGSM and PGD methods and includes an adversarial training defense.

Project Structure
DCNN Training & Evaluation: A Deep CNN model is trained on CIFAR-10, with performance evaluations on the test set.
Adversarial Attacks: Implement FGSM and PGD attacks and compare the performance drop at three noise magnitudes.
Defense Mechanism: Adversarial training with FGSM to improve robustness against attacks.
Vision Transformer (ViT): Train a ViT model on CIFAR-10, leveraging the self-attention mechanism for effective classification.

Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes. It is split into: 
Training Set: 50,000 images in five batches. 
Test Set: 10,000 images.

CNN Model Architecture
Layers: 2 Convolutional blocks followed by 3 fully connected layers.
Optimizer: SGD with learning rate 0.01.
Loss Function: CrossEntropyLoss.
Training Epochs: 30
Vision Transformer Model
Model Selection: vit_small_patch16_224 from timm library.
Patches: Image divided into 16x16 patches.
Training: Adam optimizer, CrossEntropyLoss, batch size 128, trained for 20 epochs.
Data Preprocessing: Resizing to 224x224, normalization, and batching.
Adversarial Attacks
FGSM Attack: Generates adversarial samples by adding perturbations in the direction of the gradient.
Tested with epsilon values: 0.001, 0.05, 0.1.
PGD Attack: Iterative adversarial attack with perturbation constraints.
Tested with epsilon values: 0.01, 0.05, 0.1.
Defense Mechanism
Adversarial training using FGSM to improve resilience. The model was tested on FGSM-perturbed data with improved test accuracy.
