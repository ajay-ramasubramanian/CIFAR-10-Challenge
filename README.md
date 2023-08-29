# Image classification on limited dataset
This  Challenge expounds multiple avenues used to perform classification on a small dataset. It has been divided into two major sections. The first section bounds the techniques under limited data, i.e, only 50 CIFAR10 images from two random classes and no external data is allowed. While the second section portrays techniques that are still bounded to 50 images from CIFAR10 but training on external datasets are allowed. The performances and trends of the methods are explored in detail.

Challenge-1:
This challenge precludes use of any external data. Objective is to try different approaches to maximize prediction accuracy on 50 images ofCIFAR10 dataset. Apart from Neural Networks, standard machine learning algorithms also perform well on image classification. Recently, machine learning algorithms like RandomForest and SVM have been increasingly used for remote sensing applications. Hence, Method-1 tries to analyze how different models fit on the CIFAR10
dataset. This approach will utilize the following models:
 - DecisionTreeClassifier
 - RandomForestClassifier 
 - Support Vector Machines
 - Logistic Regression
 - SGDClassifier
 - KNeighborsClassifier

 All the models were individually trained on the dataset to select the best hyperparameters through scikit-learn’s in-built method GridSearchCV. For the most part, the default parameters from pytorch worked the best. To get holistic results, we tried 2 approaches in the form of Boosting and Stacking

 Boosting: Boosting is an ensemble modeling technique that improves the prediction power by converting a number of weak learners to strong learners. The models used for boosting in our experiment are: RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, SGDClassifier and SVC. There are various forms of Boosting available. We discuss three well-known boosting methods.
 - AdaBoost
 - Gradient Boost
 - XGBoost(Extream Gradient Boosting)

 Stacking: Stacking (stacked generalization) is a powerful ensemble method in machine learning that combines the predictions of multiple models. The idea is to use several base models to make individual predictions, and then combine those predictions using a meta-model to make a final prediction. The models used for stacking in our experiment are: RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, KNeighborsClassifier and SVC.