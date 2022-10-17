# Multimedia Classification System
By Ignacio Arnaiz Tierraseca & Luis Daniel Casais Mezquida  
Multimedia 22/23  
Bachelor's Degree in Computer Science and Engineering, grp. 89  
Universidad Carlos III de Madrid

Movie poster classification.

**NOTE:** Input data must be provided into `/data/data.mat`. You can find the original file [here](https://drive.google.com/file/d/1hsST5203dYhF4V6aumZIQmo-elD43hs4/view?usp=sharing).

## Introduction and goals

The analysis and processing of multimedia data has been a powerful and versatile field of research for many years. One of the many applications of multimedia processing is to extract useful information that can be later used to solve classification problems.    
In this project, a movie genre recognition system will be developed. To this end, we will focus on visual features present in the movie posters, as well as on text-based characteristics from their description, synopsys, cast, etc. The database used for this project has been extracted from the well known web platform [IMDb](https://www.imdb.com/).  

The main goals of the project can be summarized in two points:
- Perform an effective visual and text-based feature extraction that properly represents the object under analysis.
- Achieve a decent classification accuracy in the task of distinguishing the movie genres of comedy and drama, based on the extracted features.

Upon the conclusion of this project, the student will have obtained a basic grasp of some of the main basic tools regarding image and text processing.

## Development
The development of the project will be split into 4 stages:
1. Visual feature extraction
2. Text-based feature extraction
3. Training and classification
4. Performance evaluation

### Stage 0: Database generation
The first step when starting a new project involving classification systems is the generation of a dataset.  
We will have two sets of data: the training set and the test set. For each set we will have a variable `X`, containing the samples (movie posters and descriptive texts) and a variable `Y`, which contains a label (a number between zero -comedy- and one -drama- that defines the genre) for each sample. 

In this system, the test samples account for 40% of all data, while the remaining 60% is used for training.

### Stage 1: Visual feature extraction
The main goal of feature extraction is to find a way to effectively represent complex data. For this stage, we will focus on the movie posters: images with size 268x182 px and 3 color channels (RGB). We can see that, even in cases like this, where image resolution is not particularly great, we need thousands of values to represent a single sample.  

We intend to represent each given sample with just 3 values. We will do so through the use of image processing tools. The trouble resides in properly selecting which features to extract and how to extract them so that they are decisive in distinguishing cinematographic genres. Therefore, we will extract features regarding (1) the changes in the color, (2) the brightness and (3) the amount of information present on the edges of the analyzed image.

#### Changes in color
When working with color features, the HSV color space is a very powerful ally. Particularly, one of its channels (Hue) will be used for this feature. We will try to figure out how high is the variability of this channel. To this end, we will make use of the concept of entropy, and the functions
`rgb2hsv` and `entropy`.

#### Brightness
Light intensity within a picture gives us a nice approximation of the content of said image. For instance, for outdoors photographs, light can give you an idea about the time the picture was captured. Here, we will be making use of Matlab’s function `mean` to extract the brightness feature as the mean intensity of the image’s Value channel (HS**V**).

#### Edges
The gradient of an image can be used, among other things, to extract information regarding the contour of the elements that are present in the image. Use the function `edge` to extract -through Sobel method- the edges of the posters. Subsequently, extract the third and last visual feature as the total amount of pixels that belong to an edge.  

At this point, you should have 3 values effectively representing each sample (a 268x182 color image representing a movie poster).  

As we have seen, our input data consists of images representing movie posters. For the selected visual features, there is no need to apply any preprocessing algorithms over the images. However, some techniques like segmentation or mathematical morphology are often used to prepare our images for a more efficient feature extraction.

### Stage 2: Text-based feature extraction
One of the main advantages of descriptor-based classification systems is that, once the features are extracted, they are but simple values representing a certain sample. For this reason, we are able to take different types of information (as long as they represent the same element) and mix them into a single classification system.  

This type of processing encompasses a field called Natural Language Processing (NLP) and will be introduced in depth during the second part of this subject. Nonetheless, we will now proceed to extract two very simple features based on the text document.  

For this first approach, we perform a few simple NLP algorithms to separate the text document into individual words through the use of the function `obtain_word_array`. Therefore, you can consider the resulting variable (words) as a vector containing the words belonging to the description of each movie.

Select each text document from the training data and extract the text-based features as the amount of words in each document and the mean length of said words, using Matlab’s functions `length` and `strlength`.

### Stage 3: Training and classification
If the feature extraction stage has been correctly implemented, you should now have a variable called features in your workspace, consisting of 960 rows -one for each training sample- and 5 columns -one for each extracted feature-. However, if we analyze the mentioned variable, we can observe that the range of values for each feature varies wildly. This could be harmful to the system, since a classification algorithm could give greater weights to certain features, making others insignificant regardless of their actual usefulness. For this reason, it is important to normalize the features before proceeding to train the classification system.

Normalize each feature separately. Remember that the formula used to normalize a vector _x_ is the following:

![normalization equation](https://latex.codecogs.com/svg.image?x_{n}&space;=&space;\frac{x&space;-&space;\mu_{x}}{\sigma_{x}})
<!-- thanks to https://www.codecogs.com/latex/eqneditor.php for the equation -->

With _μx_ and _σx_ being the mean and standard deviation of vector _x_, respectively. You may use Matlab’s functions `mean` and `std`.  
Check if the normalization was correctly performed making use of the function `check_normalization`.

Let us stop for a moment at this point and reflect on the usefulness of the extracted features. You can find a section devoted to feature visualization. Through that piece of code, you may choose to display (two at a time) the values of the features for all training samples via scatter plot. Try to display the different combinations of the first 3 features (visual features).

Once we have the normalized feature matrix, we may proceed to train the system. The main goal is to generate a model using the extracted features and the knowledge we have about each sample (the labels identifying each genre). This model must be able to classify future unknown samples into the two proposed categories (comedy and drama).

We will use a simple Gaussian classifier to train the system. As we are handling a binary classification problem (only two classes are present), we will consider one of them as the positive class (category to identify) and the other as the negative class (category to avoid). This nomenclature is quite useful when dealing with classes which are clearly exclusive classes. For instance, when analyzing medical images: cancer (positive class), no cancer (negative class). In this case, however, the choice of the positive class is merely anecdotal and, therefore, arbitrary. Here we use the function `fit_gaussian`, located in the `/lib/` directory.

Subsequently, you must analyze a movie from the training set. In order to select it, you need to add the digits of the student’s ID numbers (NIA) of all the group members. The resulting number will be the index of the movie whose poster has to be analyzed, script saved in `test_one.m`.

Our classification system is now complete. As you can see, we have trained three classification models: (1) using all extracted features, (2) using only visual features and (3) using only text-based features. However, this is useless unless we can provide any numerical confirmation that our system is working properly. To this end, we will see how our system works with samples that have never been seen before.

### Stage 4: Performance evaluation
Let us remember that, before we started processing the data, we separated it into two different sets: the training set -devoted to obtaining our classification model during the previous stages of this project- and the test set. In this stage, we will use this set to evaluate the performance of our system.

Test samples must be subject to the exact same processing that was applied to the training samples (feature extraction and normalization of the test images).

Remember that, regarding normalization, you must not recalculate the mean and standard deviation values. The values that were obtained during the previous stage must be reused now. 

Once this process is implemented, run the corresponding sections to obtain the normalized feature matrix from the test set (`features_test_n`).

Through the use of the function `predict_gaussian`, we can use the model obtained in the previous stage to predict the label of each test sample.

Now that we have the predicted label for each sample, we may evaluate the performance of the system. To this end, we will compare these predicted labels (the genres our system thinks are correct) with the originals (the true genres of the movies).

One way to get an idea of the performance of a classification system based on predictions is through the Probability of Detection and the Probability of False Alarm. These concepts are briefly reminded subsequently:
1. Probability of Detection:  
![detection equation](https://latex.codecogs.com/svg.image?\rho_{D}&space;=&space;\frac&space;{Positive\&space;samples\&space;correctly\&space;identified}{Total\&space;positive\&space;samples})
2. Probability of False Alarm:  
![false alarm equation](https://latex.codecogs.com/svg.image?\rho_{FA}&space;=&space;\frac&space;{Negative\&space;samples\&space;incorrectly\&space;identified\&space;as\&space;positive}{Total\&space;negative\&space;samples})

Compare the predicted labels from the complete model (`labels_pred`) with the true labels (`labels_true`) and compute the aforementioned metrics.  

Another way to evaluate the performance of the system is through the Area Under the Curve (AUC). The AUC is a value in the range [0.5, 1] that determines the performance of your system with different possible configurations.  

Run the section and observe the figure that is displayed.