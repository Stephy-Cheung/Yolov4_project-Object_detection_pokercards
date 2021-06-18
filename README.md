<img src="banner.png" width="600">

# Project-Yolov4_Object_detection(Pokercards)
This project is our first attempt on using Yolov4, model will be trained on a custom dataset for poker cards reading.Then, we will give game suggestion (Black Jack) base on object detection result. We are aiming at develop skills on training and fine tuning Yolov4 model on object detection, for future application.

## Table of Contents

- [Project background & aim](#Project_background_and_aim)
- [Base Model Construction](#Base_Model_Construction)
- [Data Collection & Preprocessing](#Data_Collection_&_Preprocessing)
  - [Image Labeling](#Image_Labeling)
  - [Image Augmentation](#Image_Augmentation)
- [Model Training](#Model_Training)
  - [Model Training (Model1)](#Model_Training_(Model1))
  - [Model Training (Model2)](#Model_Training_(Model2))
- [Poker Card Prediction](#Poker_Card_Prediction)
- [Model Deployment-BlackJack Strategy](#Model_Deployment-BlackJack_Strategy)
- [Application on Streamlit](#Application_on_Streamlit)
- [Challenges](#Challenges)
- [Insight](#Insight)

## Project_background_and_aim
  Yolov4 is an algorithm that uses neural networks to perform real-time object detection. The model will predict various probabilities (Object class) and bounding box (location) simultaneously. Yolo was firstly introduced by Joseph Redmon in 2016 and Yolov4 was created by Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao in 2020.  <br>
  
  We aimed at train a Yolov4 model on custom dataset for poker card reading on table, and give strategies suggestions for players in a Black Jack game.<br>
  
## Base_Model_Construction
  There are a number of object detection models available and mostly are general object detections. <br>
  Comparison of the performance (average precision = AP) of different models available online is as follows,
  <img src="02.png" width="600">
  
  For our project, our model will be trained to detect 52 distinct Poker card and the result will be used on the second step for generate game suggestion. <br>

  With the ability of high precision, multiple object detection and real-time object detection, Yolov4 would be ideal for our needs to detect multiple poker cards and on table for generate game suggestion. Yolov4 also has a good learning capabilities which we can also apply transfer learning to train the classification model on our custom dataset. <br>

## Project Approach 
<img src="01.png" width="600">

## Data_Collection_&_Preprocessing
 The class <br>

<img src="" width="600">

<img src="" width="600">

### Image_Labeling

### Image_Augmentation




 <br>
<img src="" width="600">
<img src="" width="600">

<br>
<img src="" width="600">

## Model_Training


### Model_Training_(Model1)
 <br>

<br>

<img src="" width="600">

### Model_Training_(Model2)

<img src="" width="600">
###
<img src="" width="600">

## Poker_Card_Prediction
 <br>

## Model_Deployment-BlackJack_Strategy

## Application_on_Streamlit

## Challenges
 <br>
 <br>

## Insight
