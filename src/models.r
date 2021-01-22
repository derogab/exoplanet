## install.packages("caret")
## install.packages("C50")
## install.packages("ROCR")
## install.packages("pROC")
## install.packages("e1071")
#install.packages("naivebayes")
#install.packages("RSNNS")
#install.packages("kernlab")

library(caret)
library(C50)
library(ROCR)
library(pROC)

## network = readRDS("models/nn.rds")
## bayes = readRDS("models/bayes.rds")
## svmP = readRDS("models/svm.polynomial.rds")
## svmR = readRDS("models/svm.radial.rds")

dataTrain = read.csv("datasets/out/train_pca.csv")
dataTest = read.csv("datasets/out/test_pca.csv")

# Scale datasets
dataTrain[c(2,ncol(dataTrain))] <- scale(dataTrain[c(2,ncol(dataTrain))])
dataTest[c(2,ncol(dataTest))] <- scale(dataTest[c(2,ncol(dataTest))])

# Factorize the label
dataTrain$koi_disposition = factor(dataTrain$koi_disposition)
dataTest$koi_disposition = factor(dataTest$koi_disposition)

#Formatting labels for caret
levels(dataTrain$koi_disposition) <- make.names(levels(factor(dataTrain$koi_disposition)))
levels(dataTest$koi_disposition) <- make.names(levels(factor(dataTest$koi_disposition)))

# Use ten fold cross validation
control = trainControl(method = "repeatedcv", number = 10, repeats = 3,
                       classProbs= TRUE, savePredictions = "final",
                       summaryFunction= twoClassSummary, verboseIter=TRUE)

#Train SVM model with radial kernel
svmRC = train(koi_disposition~ ., data = dataTrain, method = "svmRadial",
              metric= "ROC", trControl = control)
saveRDS(svmRC, "models/svm.radial_caret.rds")

#Train SVM model with polynomial kernel
svmPC = train(koi_disposition~ ., data = dataTrain, method = "svmPoly",
               metric = "ROC", trControl = control)
saveRDS(svmPC, "models/svm.polynomial_caret.rds")

#Train Bayes model
bayesC = train(koi_disposition~ ., data = dataTrain, method = "naive_bayes",
                metric= "ROC", trControl = control)
saveRDS(bayesC, "models/bayes_caret.rds")

#Train Neural network model
tunegrid <- expand.grid(size = c(as.integer(length(head(dataTrain))*(1/6)),
                                 as.integer(length(head(dataTrain))*(1/3)),
                                 as.integer(length(head(dataTrain))*(2/3))),
                        decay = c(0, 0.001, 0.1))
networkC = train(koi_disposition~ ., data = dataTrain, method = "nnet",
                 type = 'Classification',
                 tuneGrid = tunegrid,
                 metric = "ROC", trControl = control)
saveRDS(networkC, "models/network_caret.rds")

#networkC = readRDS("models/network_caret.rds")
#bayesC = readRDS("models/bayes_caret.rds")
#svmPC = readRDS("models/svm.polynomial_caret.rds")
#svmRC = readRDS("models/svm.radial_caret.rds")
# Test SVM with radial kernel, compute confusion matrix and performance measures
svmRC.pred = predict(svmRC, dataTest)
svmRC.probs = predict(svmRC, dataTest, type="prob")
svmRC.cm = confusionMatrix(table(svmRC.pred, dataTest$koi_disposition), mode = "everything")

# Test SVM with polynomial kernel, compute confusion matrix and performance measures
svmPC.pred = predict(svmPC, dataTest)
svmPC.probs = predict(svmPC, dataTest, type="prob")
svmPC.cm = confusionMatrix(table(svmPC.pred, dataTest$koi_disposition), mode = "everything")

# Test Bayes, compute confusion matrix and performance measures
bayesC.pred = predict(bayesC, dataTest)
bayesC.probs = predict(bayesC, dataTest, type="prob")
bayesC.cm = confusionMatrix(table(bayesC.pred, dataTest$koi_disposition), mode = "everything")

# Test Neural network, compute confusion matrix and performance measures
networkC.pred = predict(networkC, dataTest)
networkC.probs = predict(networkC, dataTest, type="prob")
networkC.cm = confusionMatrix(table(networkC.pred, dataTest$koi_disposition), mode = "everything")

#Print png for confusion matrixes
cat("confusion matrix for svm radial:\n")
svmRC.cm
png(filename="outputs/confusion_matrix_svmRadial.comparison.png")
fourfoldplot(svmRC.cm$table)
garbage <- dev.off()

cat("\nconfusion matrix for svm polynomial:\n")
svmPC.cm
png(filename="outputs/confusion_matrix_svmPolynomial.comparison.png")
fourfoldplot(svmPC.cm$table)
garbage <- dev.off()

cat("\nconfusion matrix for bayes:\n")
bayesC.cm
png(filename="outputs/confusion_matrix_bayes.comparison.png")
fourfoldplot(bayesC.cm$table)
garbage <- dev.off()

cat("\nconfusion matrix for network:\n")
networkC.cm
png(filename="outputs/confusion_matrix_network.comparison.png")
fourfoldplot(networkC.cm$table)
garbage <- dev.off()

# Print ROC for every model in the same plot
png(filename="outputs/roc_full.comparison.png")
svmRC.ROC = roc(response = dataTest$koi_disposition,
                predictor = svmRC.probs$CONFIRMED,
                levels = c("CONFIRMED", "FALSE.POSITIVE"))
plot(svmRC.ROC, type = "S", col = "green")

svmPC.ROC = roc(response = dataTest$koi_disposition,
                predictor = svmPC.probs$CONFIRMED,
                levels = c("CONFIRMED", "FALSE.POSITIVE"))
plot(svmPC.ROC, add = TRUE, col = "red")

bayesC.ROC = roc(response = dataTest$koi_disposition,
                 predictor = bayesC.probs$CONFIRMED,
                 levels = c("CONFIRMED", "FALSE.POSITIVE"))
plot(bayesC.ROC, add = TRUE, col = "blue")

networkC.ROC = roc(response = dataTest$koi_disposition,
                   predictor = networkC.probs$CONFIRMED,
                   levels = c("CONFIRMED", "FALSE.POSITIVE"))
plot(networkC.ROC, add = TRUE, col = "orange")
garbage <- dev.off()

# Print ROC for every model in the separate plots
png(filename="outputs/roc_svm_radial.comparison.png")
plot(svmRC.ROC, type = "S", col = "green")
garbage <- dev.off()
png(filename="outputs/roc_svm_polynomial.comparison.png")
plot(svmPC.ROC, type = "S", col = "red")
garbage <- dev.off()
png(filename="outputs/roc_bayes.comparison.png")
plot(bayesC.ROC, type = "S", col = "blue")
garbage <- dev.off()
png(filename="outputs/roc_network.comparison.png")
plot(networkC.ROC, type = "S", col = "orange")
garbage <- dev.off()

# Comparison between models statistics
cv.values = resamples(list(svm_radial = svmRC,
                           svm_polynomial = svmPC,
                           bayes = bayesC,
                           network = networkC))
summary(cv.values)

png(filename="outputs/dotplot.comparison.png")
dotplot(cv.values, metric= "ROC")
garbage <- dev.off()

png(filename="outputs/bwplot.comparison.png")
bwplot(cv.values, layout = c(3, 1))
garbage <- dev.off()

png(filename="outputs/splom.comparison.png")
splom(cv.values, metric="ROC")
garbage <- dev.off()

# Print models timings
cv.values$timings
