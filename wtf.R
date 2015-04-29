library("caret")
library(randomForest)
library(dplyr)
library("doMC")

# I want to leave some cores for other things.
doMC::registerDoMC(cores=detectCores() -3 )

inputData <- read.csv("train.csv")
testData <- read.csv("test.csv")

# Cleaning and tweaking
inputData$label <- as.factor(inputData$label)

# Partition the training data
randomSelect <- createDataPartition(inputData$label, p=0.7, list=F)
modelTrain <- inputData[randomSelect, ]
modelTest <- inputData[-randomSelect, ]

# Consider removing columns with near zero variance
nzv <- nearZeroVar(modelTrain)
modelTrain <- modelTrain[,-nzv]

# Super simple analysis. Make a table of outcomes and use the most
# common as my guess across everything
prop.table(table(modelTrain$label))
static <- rep(1, 12596)
static <- factor(static, levels=0:9)
confusionMatrix(static, modelTest$label) # 11% accuracy! Yes!

if(!file.exists("rfModel.Rda")) {
  rfStart <- Sys.time()
  rfModel <- train(label ~ ., data=modelTrain, method="rf", verbose=F)
  rfEnd <- Sys.time()
  save(rfModel, file="rfModel.Rda")
}

if(!file.exists("rfRegular.Rda")) {
  randStart <- Sys.time()
  rfRegular <- randomForest(modelTrain[,-1], modelTrain$label)
  randEnd <- Sys.time()
  save(rfRegular, file="rfRegular.Rda")
}

if(!file.exists("knnModel.Rda")) {
  knnStart <- Sys.time()
  knnModel <- train(label ~ ., data=modelTrain, method="knn")
  knnEnd <- Sys.time()
  save(knnModel, file="knnModel.Rda")
}

if(!file.exists("nnModel.Rda")) {
  nnStart <- Sys.time()
  nnModel <- train(label ~ ., data=modelTrain, method="nnet")
  nnEnd <- Sys.time()
  save(nnModel, file="nnModel.Rda")
}

submitData <- data.frame(ImageID=seq(1,length(knnPredictions)), Label=knnPredictions)
write.csv(submitData, file="submitdata.csv", quote = F, row.names = F)