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
- [Result & Prediction](#Result_&_Prediction)
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
 Yolo v4 was trained with 80 classes but nor of them are included poker cards labels. Therefore, the model needs to be trained on custom dataset for our project. <br>

 To train a Yolo v4 model, we have to feed the model with our target class label and respective location on the image. Since no preprocessed dataset is available on online, we started our project at prepare our own dataset (image and labels) for training.<br>

## Image_Labeling
 Our first batch image collection consist of 165 images and second batch consist of 297 images. Using the community tools, 'labelImg', we created 52 unique class labes and labeled all the target classes on every image one by one. The tools will return a txt file which containing the class, the coordinates and the size of each label respectively. <br>

 The video and image below showcases the labeling process and result in details, <br>
<video>

<img src="03.png" width="600">

## Image_Augmentation
 Image augmentation was applied to expand the dataset and create variation to the images to improve the ability of the model to generalizei in detection. Rotation, shear, exposure and noise were randomly applied on different image and tripled the size of the dataset (around 1.2K images). <br>
 
 The below image is one of the example of original image and processed image.
<img src="06.png" width="600">

## Model_Training
 Two models were trained in this project. <br>
 
## Model_Training_(Model1)
 Model1 was trained with first batch of images (Training: 133 images; Testing: 32 images).  <br>
 
 After around 1000 epochs of training, the model can detect close up image of poker cards but not perform well if the background is messy or poker image is rotated and sheared.<br>
 
 <img src="04.png" width="600">

 Therefore, we have two approaches to continue the project. <br>

 For Model1, we continue the model training with additional images with different backgrounds and rotated images to help the model more generalize.

 On the other hand, we collected more images online and applied image augmentation to expand the dataset to start the training of Model2. 

## Model_Training_(Model2)
 Model2 was trained with the final data set of around 1.2K images. (Training: 1100 images; Testing: 108 images)<br>
 Image Augmentation applied and we aimed at having a more sensitive and generalized model. <br>
<img src="07.png" width="600">

## Result_&_Prediction
 After few more thousands of epoches training, the final mAP(mean average precision) at IOU 0.5 (Intersection over Union) of Model1 and Model2 is as below. <br>
 Model1: 65.34%<br>
 Model2: 51.22% <br>
 
 Although Model2 seems slightly underperform compare with Model1, we would like to tested the model with different scenarios to better picture their strength and performance. Focusing on the accuracy of locating the card and the accuracy of card reading. <br>

 <img src="09.png" width="600">
 <img src="10.png" width="600">
 Both models perform similar in detect sheared Poker images. However, ever Model can detect image closed to the chip, the detection is incorrect.  <br>
 <img src="11.png" width="600"> 
 <img src="12.png" width="600">
 <img src="13.png" width="600">
 <img src="14.png" width="600">
 For close up images, both models perform well in image location and image classification. Minor underperform of Model 2 as it might wrongly detected the inverted 4 of Club as Ace of Club, due to the similarity of the character 4 and A.<br>
 <img src="15.png" width="600">
 <img src="16.png" width="600">
 For messy background images, Model2 can detect more target on the image.<br>

 #### In conclude, Model1 perform better on clear and well-defined images as the training dataset while Model2 are more generalized for image with different angle and more noise.<br>

## Model_Deployment-BlackJack_Strategy

### **Blackjack Strategy based on Detection Result**
The detection result can be output in a dictionary-like string as below, which can be parsed easily.<br>

```
[
{
 "frame_id":1, 
 "filename":"data/poker.jpg", 
 "objects": [ 
  {"class_id":17, "name":"KC", "relative_coordinates":{"center_x":0.787017, "center_y":0.571584, "width":0.305575, "height":0.481816}, "confidence":0.983804}, 
  {"class_id":16, "name":"10D", "relative_coordinates":{"center_x":0.207783, "center_y":0.720466, "width":0.221507, "height":0.189736}, "confidence":0.994300}, 
  {"class_id":1, "name":"JS", "relative_coordinates":{"center_x":0.364248, "center_y":0.562933, "width":0.122483, "height":0.663116}, "confidence":0.997848}
 ] 
}
]
```

To distinguish dealer's card and player's card, we used y coordinate of the card location as the guideline. As the image/photo is taken from the player's perspective, the dealer's cards are always within the upper part of the image, while the player's cards are within the lower part.<br>

Sometimes, the same card might be detected multiple times if it is shown fully (ranks and suits are printed twice on the same card). To prevent this, we extract a unique list of card from the detection results, as there must be no duplication for a standard 52-card deck.<br>

The strategy of playing Blackjack is a simple probability problem, which can be resolved easily by simulating all possible outcomes. To increase the speed of the program, we hard-coded the calculation results which the program can refer to, instead of doing the calculation every time.<br>

<img src="17,png" width="600">

<img src="18.png" width="600">



## Application_on_Streamlit


<img src="" width="600">

## Challenges
 <br>
 <br>

## Insight
