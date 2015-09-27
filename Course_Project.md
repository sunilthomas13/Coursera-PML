Practical Machine learning: Course Project
==========================================

Sunil Thomas


##Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. 
These type of devices are part of the quantified self movement â 
a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

### Settings

```r
echo = TRUE  # Always make code visible
options(scipen = 1)  # Turn off scientific notations for numbers
```

### Reading and understanding data

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(knitr)
```

```
## Warning: package 'knitr' was built under R version 3.2.2
```

```r
set.seed(333)


training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

dim(training); dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```


### Cleaning and pre-processing data

*Cleaning the training set

```r
training <- training[, colSums(is.na(training)) == 0] 
testing <- testing[, colSums(is.na(testing)) == 0] 

classe <- training$classe
trainRemove <- grepl("^X|timestamp|window", names(training))
training <- training[, !trainRemove]
train_clean <- training[, sapply(training, is.numeric)]
train_clean$classe <- classe
```

*Cleaning the testing set

```r
testRemove <- grepl("^X|timestamp|window", names(testing))
testing <- testing[, !testRemove]
test_clean <- testing[, sapply(testing, is.numeric)]
```

### Modeling
*Partition training data and for training the model use 10 fold cross validation


```r
Trainpart <- createDataPartition(train_clean$classe, p=0.65, list=F)
Train_train <- train_clean[Trainpart, ]
Train_test <- train_clean[-Trainpart, ]

controlrfclf <- trainControl(method="cv", 10)
rfclf <- train(classe ~ ., data=Train_train, method="rf", trControl=controlrfclf, ntree=250)
rfclf
```

```
## Random Forest 
## 
## 12757 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 11481, 11482, 11481, 11481, 11481, 11481, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9908283  0.9883971  0.001993090  0.002521885
##   27    0.9918476  0.9896872  0.002456205  0.003106585
##   52    0.9826761  0.9780835  0.003776049  0.004777073
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
rfclf_predict <- predict(rfclf, Train_test)
```

### Evaluation

* Get accuracy and out of sample error

```r
confusionMatrix(Train_test$classe, rfclf_predict)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1950    2    1    0    0
##          B    9 1319    0    0    0
##          C    0   11 1184    2    0
##          D    0    1   14 1109    1
##          E    0    1    1    5 1255
## 
## Overall Statistics
##                                           
##                Accuracy : 0.993           
##                  95% CI : (0.9907, 0.9948)
##     No Information Rate : 0.2854          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9912          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9954   0.9888   0.9867   0.9937   0.9992
## Specificity            0.9994   0.9984   0.9977   0.9972   0.9988
## Pos Pred Value         0.9985   0.9932   0.9891   0.9858   0.9945
## Neg Pred Value         0.9982   0.9973   0.9972   0.9988   0.9998
## Prevalence             0.2854   0.1943   0.1748   0.1626   0.1830
## Detection Rate         0.2840   0.1921   0.1725   0.1615   0.1828
## Detection Prevalence   0.2845   0.1934   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9974   0.9936   0.9922   0.9955   0.9990
```

```r
accuracy <- postResample(rfclf_predict, Train_test$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9930080 0.9911547
```

```r
error <- 1 - as.numeric(confusionMatrix(Train_test$classe, rfclf_predict)$overall[1])
error
```

```
## [1] 0.006991988
```

### Prediction


```r
result <- predict(rfclf, test_clean[, -length(names(test_clean))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Notes
Why did I choose Random Forest over Decision trees?
Decision trees usually over-fit the training data and this leads to high variance predictions. Random forests overcome this particular problem.
Considering that the error is just around 0.006%, this was a good decision. 
