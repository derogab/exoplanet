library(caret)
library(C50)
library(ROCR)
library(pROC)
library(e1071)
library(neuralnet)

# Split function
split.data = function(data, p = 0.7, s = 1){
    set.seed(s)
    index = sample(1:dim(data)[1])
    train = data[index[1:floor(dim(data)[1] * p)], ]
    test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
    return(list(train=train, test=test))
}

nn10cv = function(data){
    cv.error = NULL
    for(i in 1:10){
        index <- sample(1:nrow(data),round(0.9*nrow(data)))
        train.cv <- data[index,]
        test.cv <- data[-index,]
        
        nn <- neuralnet(koi_disposition ~ .,data=train.cv,hidden=c(2,2),linear.output=T)
        
        pr.nn <- compute(nn,test.cv[,1:13])
        pr.nn <- pr.nn$net.result*(max(data$koi_disposition)-min(data$koi_disposition))+min(data$koi_disposition)
        
        test.cv.r <- (test.cv$koi_disposition)*(max(data$koi_disposition)-min(data$koi_disposition))+min(data$koi_disposition)
        
        cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
    }
}


network = readRDS("models/nn.rds")
bayes = readRDS("models/bayes.rds")
svmP = readRDS("models/svm.polynomial.rds")
svmR = readRDS("models/svm.radial.rds")

dataTrain = read.csv("datasets/tmp/train_pca.csv")
dataTrain = dataTrain[dataTrain$koi_disposition != "CANDIDATE",]
dataTest = read.csv("datasets/tmp/test_pca.csv")
dataTest = dataTest[dataTest$koi_disposition != "CANDIDATE",]
dataFull = rbind(dataTrain, dataTest)
                                        # Scale datasets
dataTrain[c(2,ncol(dataTrain))] <- scale(dataTrain[c(2,ncol(dataTrain))])
dataTest[c(2,ncol(dataTest))] <- scale(dataTest[c(2,ncol(dataTest))])
dataFull[c(2,ncol(dataFull))] <- scale(dataFull[c(2,ncol(dataFull))])
                                        # Factorize the label
dataTrain$koi_disposition = factor(dataTrain$koi_disposition)
dataTest$koi_disposition = factor(dataTest$koi_disposition)
## setup for 10-fold cross-validation
## control = trainControl(method = "repeatedcv", number = 10,repeats = 3, 
##                        classProbs= TRUE, summaryFunction= twoClassSummary)
## non funziona
bayesCV = tune.naiveBayes(koi_disposition ~ .,
                          data = dataFull,
                          laplace = c(0, 1, 2, 3),
                          prob=TRUE,
                          tunecontrol=tune.control(cross=10))

svmRCV = tune.svm(koi_disposition ~ .,
                 data = dataFull,
                 kernel = 'radial',
                 prob=TRUE,
                 tunecontrol=tune.control(cross=10))

svmPCV = tune.svm(koi_disposition ~ .,
                 data = dataFull,
                 kernel = 'polynomial',
                 grades = c(2,3,4,5,6,7,8,9,10,11),
                 coef0 = c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                 prob=TRUE,
                 tunecontrol=tune.control(cross=10))

## si ribella
networkCV = nn10cv(dataFull)

network.pred = predict(network, dataTest, probability = TRUE)
bayes.pred = predict(bayes, dataTest, type = "raw")
svmP.pred = predict(svmP, dataTest, probability = TRUE)
svmR.pred = predict(svmR, dataTest, probability = TRUE)
colnames(network.pred) <- c("CONFIRMED", "FALSE POSITIVE")

network.ROC = roc(response = dataTest$koi_disposition, 
                  predictor = network.pred[,1],
                  levels = c("CONFIRMED", "FALSE POSITIVE"))
dev.new() 
plot(network.ROC, type="S", col="green")

bayes.ROC = roc(response = dataTest$koi_disposition, 
                predictor = bayes.pred[,1],
                levels = c("CONFIRMED", "FALSE POSITIVE"))

plot(bayes.ROC, add=TRUE, col="red")

pred.probP = attr(svmP.pred, "probabilities") 
pred.to.rocP = pred.probP[, 1] 
svmP.ROC = roc(response = dataTest$koi_disposition, 
               predictor = pred.to.rocP,
               levels = c("CONFIRMED", "FALSE POSITIVE"))

plot(svmP.ROC, add=TRUE, col="blue")

pred.probR = attr(svmR.pred, "probabilities") 
pred.to.rocR = pred.probR[, 1] 
svmR.ROC = roc(response = dataTest$koi_disposition, 
               predictor = pred.to.rocR,
               levels = c("CONFIRMED", "FALSE POSITIVE"))

plot(svmR.ROC, add=TRUE, col="orange")


#########################
## dev.set(dev.cur())  ##
## dev.new()           ##
#########################

cv.values= resamples(list(network = network, bayes = bayes,
                          svm_poly = svmP, svm_radial = svmR))
summary(cv.values)