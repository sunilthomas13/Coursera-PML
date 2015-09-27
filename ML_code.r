library(caret)
library(rpart)
library(randomForest)
library(knitr)

set.seed(333)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

dim(training); dim(testing)

training <- training[, colSums(is.na(training)) == 0] 
testing <- testing[, colSums(is.na(testing)) == 0] 

classe <- training$classe
trainRemove <- grepl("^X|timestamp|window", names(training))
training <- training[, !trainRemove]
train_clean <- training[, sapply(training, is.numeric)]
train_clean$classe <- classe

testRemove <- grepl("^X|timestamp|window", names(testing))
testing <- testing[, !testRemove]
test_clean <- testing[, sapply(testing, is.numeric)]

Trainpart <- createDataPartition(train_clean$classe, p=0.65, list=F)
Train_train <- train_clean[Trainpart, ]
Train_test <- train_clean[-Trainpart, ]

controlrfclf <- trainControl(method="cv", 10)
rfclf <- train(classe ~ ., data=Train_train, method="rf", trControl=controlrfclf, ntree=250)
rfclf

rfclf_predict <- predict(rfclf, Train_test)

confusionMatrix(Train_test$classe, rfclf_predict)
accuracy <- postResample(rfclf_predict, Train_test$classe)
accuracy
error <- 1 - as.numeric(confusionMatrix(Train_test$classe, rfclf_predict)$overall[1])
error

result <- predict(rfclf, test_clean[, -length(names(test_clean))])
result

