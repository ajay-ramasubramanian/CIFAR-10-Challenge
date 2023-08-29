# Image classification on limited dataset
- The report explores multiple methods for performing image classification on a small dataset of 50 images from CIFAR-10 using neural networks and machine learning algorithms.

- Challenge 1 is bounded to only using the 50 CIFAR-10 images with no external data. Method 1 uses standard ML algorithms like Random Forest, Decision Tree, SVM etc. Method 2 experiments with custom neural network architectures of varying depths and hyperparameters. Method 3 uses a larger VGG-style network with regularization techniques.

- Challenge 2 allows usage of external datasets for pre-training, as long as they are not from the CIFAR family. It pre-trains models like ResNet, AlexNet on large datasets like ImageNet and STL-10, then fine-tunes them on the 50 CIFAR-10 images.

- Results show simpler 3 layer CNNs perform best on the small dataset to avoid overfitting. AdaDelta optimizes best. Mish and SiLU activations outperform ReLU. Ensembles like boosting/stacking improve accuracy.

- Pre-training on large datasets like ImageNet before fine-tuning significantly increases validation accuracy compared to training from scratch, due to learning generalized features.

- The report analyzes the various methods through evaluation metrics and discusses the experimental insights and literature reviewed for the techniques used. It systematically explores how to maximize classification on limited labeled data.