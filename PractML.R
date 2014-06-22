## set working directory 

setwd('C:/Research/Self-Development/Data-Science/8_Practical_Machine_Learning/cw')
## Read data, handle whitespace ...
training <- read.csv('data/pml-training.csv',header=TRUE, 
                     na.strings=c("NA","NaN",'',' '),stringsAsFactors=FALSE)

## Simple function to count missing values  
countMissings <- function (df) {
        
        #get missing values per column
        colIndex <- c(1:ncol(df))
        dfMissings <- data.frame(colIndex)
        dfMissings$NoOfMissings <- 0
        colnames(dfMissings) <- c("colIndex", "NoMissings")
        
        for (i in 1: ncol(df)) {
                
                dfMissings[i,2]= sum(is.na(df[,i])) 
                
        }
        ## return data frame with colIndex and number of missing Values
        dfMissings   
}


## missingVals <- countMissings(training) ## test above function


## This function aims to tidy up the data and 
## identify the columns with more than 95% of values missing 

getColsToDrop <- function (df,precent) {
        #get missing values per column
        colIndex <- c(1:ncol(df))
        dfMissings <- data.frame(colIndex)
        dfMissings$NoOfMissings <- 0
        colnames(dfMissings) <- c("colIndex", "NoMissings")

        for (i in 1: ncol(df)) {

                dfMissings[i,2]= sum(is.na(df[,i])) 
             
        }
        dfMissings <- dfMissings[dfMissings$NoMissings/nrow(df)>precent,]
        ## column names with 95% missing values 
        dorppedNames <- colnames(df[,dfMissings$colIndex])
        
}

## subset and drop the above columns 
cleanData <- function (df,percent) {
        df <- df[, !(names(df)%in% getColsToDrop(df,percent))]
}

## Remove features with more than 95% missing values 
training <- cleanData(training,0.95)

## Drop the first 7 columns of the updated training set 
training <- training[, !(names(training)%in% colnames(training[,c(1:7)]))]

## Checking for important features using features corelations 

corMat <- abs(cor(training[,-53]))
diag(corMat) <- 0
corMat <- which(corMat > 0.85,arr.ind=T)
#impVars<- names(training)[c(unique(corMat[,1]), unique(corMat[,2]))]
impVars<- c(unique(corMat[,1]), unique(corMat[,2]),53)

## only consider important features identified in previous step
training <- training[,c(impVars)]
training$classe <- factor(training$classe)
## check for missing values in the filered data set
missingVals <- countMissings(training)

## require randomForest Library
library(randomForest);library(caret);library(kernlab);


## partition the data 
inTrain <- createDataPartition(y=training$classe, p=0.6, 
                                 list=FALSE)

trainingSet <- training[inTrain,]
testingSet <- training[-inTrain,]


## Traing the rforest
set.seed(1431)
rf <- randomForest(classe ~ ., data=trainingSet,importance=TRUE,
                   ntree=500, proximity=TRUE,keep.forest=TRUE)

validatePred<- predict(rf, testingSet);
testingSet$predRight <- validatePred==testingSet$classe
table(validatePred, testingSet$classe)

## crossvalidation

importance(rf)## importance of each predictor
varImpPlot(rf)
plot( importance(rf), lty=2, pch=16)

###########################################

## Read testing, handle whitespace ...
testing <- read.csv('data/pml-testing.csv',header=TRUE, 
                     na.strings=c("NA","NaN",'',' '),stringsAsFactors=FALSE)
## apply the same changes on the testing data 

## Remove features with more than 95% missing values 
testing <- cleanData(testing,0.95)

## Drop the first 7 columns of the updated training set 
testing <- testing[, !(names(training)%in% colnames(training[,c(1:7)]))]

testPred<- predict(rf, testing);
testing$predicted <- testPred

results <- testing[,c('problem_id', 'predicted')]
results<- table(testPred, testing$problem_id)

## Function to output results (adopted from DS specialisation)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testing$predicted)
