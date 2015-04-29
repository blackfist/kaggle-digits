library("caret")
library(randomForest)
library(dplyr)

inputData <- read.csv("train.csv")
testData <- read.csv("test.csv")

# Cleaning and tweaking
inputData$label <- as.factor(inputData$label)

# Consider removing columns with near zero variance
# nzv <- nearZeroVar(training)
# training <- training[,-nzv]

# Partition the training data
randomSelect <- createDataPartition(inputData$label, p=0.7, list=F)
modelTrain <- inputData[randomSelect, ]
modelTest <- inputData[-randomSelect, ]

# Super simple analysis. Make a table of outcomes and use the most
# common as my guess across everything
prop.table(table(modelTrain$label))
static <- rep(1, 12596)
static <- factor(static, levels=0:9)
confusionMatrix(static, modelTest$label) # 11% accuracy! Yes!

# first our baseline. Train the random forest without multicores or caret
cat("baseline. randomforest no cores no caret")
system.time(rfModel <- randomForest(modelTrain[,-1], modelTrain$label))

# now the same thing in caret
cat("randomforest with caret. No cores")
system.time(rfModel <- train(label ~ ., data=modelTrain, method="rf", verbose=F))


# If we have access to a lot of cores then let's use them
library(doMC)
doMC::registerDoMC(cores=detectCores() -1 )

cat("random forest with caret and cores")
system.time(rfModel <- train(label ~ ., data=modelTrain, method="rf", verbose=F))


