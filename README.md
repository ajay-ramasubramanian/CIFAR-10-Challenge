# Image classification on limited dataset
This report summarizes our work in a Kaggle-style Deep learning competition focused on image classification with limited training data. The goal of the competition was to classify images using only 50 training samples from the CIFAR-10 dataset. We were tasked with exploring various neural network architectures and machine learning algorithms to maximize classification accuracy on this small sample. Through rigorous experimentation and systematic evaluation of different modeling approaches, we were able to achieve the second place ranking in the competition. We employed techniques such as data augmentation, transfer learning, model ensembling, activation function selection, and regularization to carefully optimize performance. This report details our full methodology, including pre-processing, model development, training procedures, results analysis, and lessons learned. In the end, we were able to demonstrate state-of-the-art methods for tackling the challenging problem of learning from extremely small labeled datasets.

Challenge 1 Method 1:

Models tested: Decision Tree, Random Forest (with 10,50,100 estimators), SVM (linear, RBF, polynomial kernels), Logistic Regression Grid search used to tune hyperparameters like max_depth, criterion, kernel AdaBoost, Gradient Boosting, XGBoost used for boosting with n_estimators=10 Stacking ensemble combined predictions from RandomForest, SVC, Logistic Regression, Decision Tree
Metrics: Accuracy, AUC-ROC, precision, recall averaged over 25 trials
RandomForest and SVC performed best as individual models at ~70% accuracy
LogisticRegression best for AdaBoost at 70% accuracy, SVC best for Gradient Boost at 68%

Challenge 1 Method 2:

5 base models with depths 3,4,6 conv layers and 2 fc layers tested AdaDelta, Adam, SGD optimizers compared, AdaDelta worked best ReLU, Mish, SiLU activations - Mish and SiLU improved on ReLU Models finessed with batchnorm, dropout, weight decay as regularizers Learning rate schedulers compared, MultiStepLR performed best model-3 with 3conv-2fc, Adam optimizer and SiLU activation achieved 79.28% accuracy

Challenge 1 Method 3:

VGG16-style network with max norm, adversarial regularization Mish and SiLU continued outperforming ReLU Dropblock showed better regularization than dropout Model-1 with max norm and SiLU achieved top accuracy of 79.24% Higher capacity meant 2x longer training time than baseline

Challenge 2: Classification using external dataset pre-training

Method 1: Pre-training large models on external datasets

Pre-training on ImageNet before fine-tuning significantly boosted accuracy ResNet and AlexNet achieved up to 92% accuracy after ImageNet pre-training STL-10 dataset provided less benefit due to limited number of classes This demonstrated ability to leverage large unlabeled datasets

In summary, each challenge methodically explored techniques like architecture design, hyperparameters, activations, regularization and transfer learning to optimize performance on the limited CIFAR-10 sample.