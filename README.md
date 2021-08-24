# Portable-Image-Classifier

Aim: Create a Portable Image Classifier using tensorflow.


Theory:
Classification: In the process of data mining, large data sets are first sorted, then patterns are identified and relationships are established to perform data analysis and solve problems. Classification: It is a Data analysis task, i.e. the process of finding a model that describes and distinguishes data classes and concepts. Classification is the problem of identifying to which of a set of categories (subpopulations), a new observation belongs to, on the basis of a training set of data containing observations and whose category membership is known.


Applications of Classification: ● Credit Card Fraud Detection:You have to Classify the transactions into valid and fraud cases so that the customers of credit card companies are not charged for items that they did not purchase ● Fake News Detection: With the growth of social media platforms, the spread of fake news is also very rapid. The task is to use a text classification approach to create an ML model that will differentiate between real and fake news. ● Email Spam Classification:Email Spam Detection is perhaps one of the most popular document classification tasks for beginners. This dataset has two columns. The first one is labeled either or spam which is a fancy way of saying whether the email is spam or not. The next columns contain email text based on which we will be classifying our emails. So, the task in this dataset will be to classify the emails into spam or not spam.


Uses of Classification :
● Mining Based Methods are cost effective and efficient ● Helps in identifying criminal suspects ● Helps in predicting risk of diseases ● Helps Banks and Financial Institutions to identify defaulters so that they may approve Cards, Loan, etc.
Algorithm: VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition''. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous models submitted to ILSVRC-2014. It makes an improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using the NVIDIA Titan Black GPU. Dataset: ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The images were collected from the web and labeled by human labelers using Amazon’s Mechanical Turk crowd-sourcing tool. Starting in 2010, as part of the Pascal Visual Object Challenge, an annual competition called the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held. ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories. At all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images. ImageNet consists of variable-resolution images. Therefore, the images have been down-sampled to a fixed resolution of 256×256. Given a rectangular image, the image is rescaled and cropped out the central 256×256 patch from the resulting image. Tensorflow: TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. Tensorflow is a symbolic math library based on dataflow and differentiable programming.


Implementation Requirements :
● Python
● Tensorflow
● Keras
● numpy
