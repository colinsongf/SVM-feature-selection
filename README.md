------------------------------------------------------------------
Support Vector Machines (SVM) feature selection
------------------------------------------------------------------

This  Python code implements a feature selection method, based on the weights assigned by the SVM during the training phase.
The feature are selected based on a threshold which can vary from 0 to 1.

	- svm_feature_selection.py 

This graph shows how the accuracy assessed on a LOO CV test vary on the SVM weights threshold. The annotations represents the number of features selected for a certain threshold.

![alt tag](https://raw.githubusercontent.com/giangi023/SVM-feature-selection/master/svm_feature_selection.png)
