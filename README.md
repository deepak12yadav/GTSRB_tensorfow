# GTSRB_tensorfow
It was a german benchmark competition for traffic sign recognition.

Dataset::http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
The dataset contains 39209 examples for training .
But i have split the dataset in two parts.One for training and other for evaluation.
Training dataset :30000
Evaluation dataset:9209



Model used :http://people.idsia.ch/~juergen/nn2012traffic.pdf (MCDNN)

Code used ::cifar10 code available in tendorflow example (just for the educational purpose )
Reference::https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image/cifar10

A new file readTrafficSigns.py file is made to convert the GTSRB dataset into the cifar10 format so that the code can be used 
small changes are made in cifar10 code and the model is changed as per convinience .

Run cifar_train.py for training
After running cifar_train.py run cifar10_eval.py from the same address.
