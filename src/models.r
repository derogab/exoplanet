## Install packages
install.packages("caret")
install.packages("C50")
install.packages("ROCR")
install.packages("pROC")
install.packages("e1071")

## Install packages for caret model
install.packages("naivebayes")
install.packages("RSNNS")
install.packages("kernlab")
install.packages("nnet")

## LOad libraries
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


# Invert factor levels (CONFIRMED <--> FALSE POSITIVE)
# Trick to have plots with FALSE POSITIVE as positive target
dataTrain$koi_disposition <- factor(dataTrain$koi_disposition, levels=rev(levels(dataTrain$koi_disposition)))
dataTest$koi_disposition <- factor(dataTest$koi_disposition, levels=rev(levels(dataTest$koi_disposition)))

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
networkC = train(koi_disposition~ ., data = dataTrain, method = "nnet",
                 type = 'Classification',
                 metric = "ROC", trControl = control)
saveRDS(networkC, "models/network_caret.rds")

networkC = readRDS("models/network_caret.rds")
bayesC = readRDS("models/bayes_caret.rds")
svmPC = readRDS("models/svm.polynomial_caret.rds")
svmRC = readRDS("models/svm.radial_caret.rds")
# Test SVM with radial kernel, compute confusion matrix and performance measures
svmRC.pred = predict(svmRC, dataTest)
svmRC.probs = predict(svmRC, dataTest, type="prob")
svmRC.cm = confusionMatrix(table(svmRC.pred, dataTest$koi_disposition), mode = "everything", positive = "CONFIRMED")
svmRC.cmN = confusionMatrix(table(svmRC.pred, dataTest$koi_disposition), mode = "everything", positive = "FALSE.POSITIVE")

# Test SVM with polynomial kernel, compute confusion matrix and performance measures
svmPC.pred = predict(svmPC, dataTest)
svmPC.probs = predict(svmPC, dataTest, type="prob")
svmPC.cm = confusionMatrix(table(svmPC.pred, dataTest$koi_disposition), mode = "everything", positive = "CONFIRMED")
svmPC.cmN = confusionMatrix(table(svmPC.pred, dataTest$koi_disposition), mode = "everything", positive="FALSE.POSITIVE")

# Test Bayes, compute confusion matrix and performance measures
bayesC.pred = predict(bayesC, dataTest)
bayesC.probs = predict(bayesC, dataTest, type="prob")
bayesC.cm = confusionMatrix(table(bayesC.pred, dataTest$koi_disposition), mode = "everything", positive = "CONFIRMED")
bayesC.cmN = confusionMatrix(table(bayesC.pred, dataTest$koi_disposition), mode = "everything", positive="FALSE.POSITIVE")

# Test Neural network, compute confusion matrix and performance measures
networkC.pred = predict(networkC, dataTest)
networkC.probs = predict(networkC, dataTest, type="prob")
networkC.cm = confusionMatrix(table(networkC.pred, dataTest$koi_disposition), mode = "everything", positive = "CONFIRMED")
networkC.cmN = confusionMatrix(table(networkC.pred, dataTest$koi_disposition), mode = "everything", positive="FALSE.POSITIVE")

#Print png for confusion matrixes
cat("Confusion matrix for svm radial for class \"CONFIRMED\":\n")
svmRC.cm
cat("Confusion matrix for svm radial for class \"FALSE POSITIVE\":\n")
svmRC.cmN
png(filename="outputs/confusion_matrix_svmRadial.comparison.png")
fourfoldplot(svmRC.cmN$table)
garbage <- dev.off()

cat("\nConfusion matrix for svm polynomial for class \"CONFIRMED\":\n")
svmPC.cm
cat("\nConfusion matrix for svm polynomial for class \"FALSE POSITIVE\":\n")
svmPC.cmN
png(filename="outputs/confusion_matrix_svmPolynomial.comparison.png")
fourfoldplot(svmPC.cmN$table)
garbage <- dev.off()

cat("\nConfusion matrix for bayes for class \"CONFIRMED\":\n")
bayesC.cm
cat("\nConfusion matrix for bayes for class \"FALSE POSITIVE\":\n")
bayesC.cmN
png(filename="outputs/confusion_matrix_bayes.comparison.png")
fourfoldplot(bayesC.cmN$table)
garbage <- dev.off()

cat("\nConfusion matrix for network for class \"CONFIRMED\":\n")
networkC.cm
cat("\nConfusion matrix for network for class \"FALSE POSITIVE\":\n")
networkC.cmN
png(filename="outputs/confusion_matrix_network.comparison.png")
fourfoldplot(networkC.cmN$table)
garbage <- dev.off()

# Print ROC for every model in the same plot using CONFIRMED label
png(filename="outputs/roc_full.comparison_CONFIRMED.png")
svmRC.ROC_CONFIRMED = roc(response = dataTest$koi_disposition,
                predictor = svmRC.probs$CONFIRMED,
                levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(svmRC.ROC_CONFIRMED, type = "S", col = "green")

svmPC.ROC_CONFIRMED = roc(response = dataTest$koi_disposition,
                predictor = svmPC.probs$CONFIRMED,
                levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(svmPC.ROC_CONFIRMED, add = TRUE, col = "red")

bayesC.ROC_CONFIRMED = roc(response = dataTest$koi_disposition,
                 predictor = bayesC.probs$CONFIRMED,
                 levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(bayesC.ROC_CONFIRMED, add = TRUE, col = "blue")

networkC.ROC_CONFIRMED = roc(response = dataTest$koi_disposition,
                   predictor = networkC.probs$CONFIRMED,
                   levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(networkC.ROC_CONFIRMED, add = TRUE, col = "orange")
garbage <- dev.off()

# Print ROC for every model in the same plot using FALSE.POSITIVE label
png(filename="outputs/roc_full.comparison_FALSEPOSITIVE.png")
svmRC.ROC_FALSEPOSITIVE = roc(response = dataTest$koi_disposition,
                predictor = svmRC.probs$FALSE.POSITIVE,
                levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(svmRC.ROC_FALSEPOSITIVE, type = "S", col = "green")

svmPC.ROC_FALSEPOSITIVE = roc(response = dataTest$koi_disposition,
                predictor = svmPC.probs$FALSE.POSITIVE,
                levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(svmPC.ROC_FALSEPOSITIVE, add = TRUE, col = "red")

bayesC.ROC_FALSEPOSITIVE = roc(response = dataTest$koi_disposition,
                 predictor = bayesC.probs$FALSE.POSITIVE,
                 levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(bayesC.ROC_FALSEPOSITIVE, add = TRUE, col = "blue")

networkC.ROC_FALSEPOSITIVE = roc(response = dataTest$koi_disposition,
                   predictor = networkC.probs$FALSE.POSITIVE,
                   levels = c("FALSE.POSITIVE", "CONFIRMED"))
plot(networkC.ROC_FALSEPOSITIVE, add = TRUE, col = "orange")
garbage <- dev.off()

# Print ROC for every model in the separate plots for CONFIRMED
png(filename="outputs/roc_svm_radial.comparison_CONFIRMED.png")
plot(svmRC.ROC_CONFIRMED, type = "S", col = "green", print.auc=TRUE)
garbage <- dev.off()
png(filename="outputs/roc_svm_polynomial.comparison_CONFIRMED.png")
plot(svmPC.ROC_CONFIRMED, type = "S", col = "red", print.auc=TRUE)
garbage <- dev.off()
png(filename="outputs/roc_bayes.comparison_CONFIRMED.png")
plot(bayesC.ROC_CONFIRMED, type = "S", col = "blue", print.auc=TRUE)
garbage <- dev.off()
png(filename="outputs/roc_network.comparison_CONFIRMED.png")
plot(networkC.ROC_CONFIRMED, type = "S", col = "orange", print.auc=TRUE)
garbage <- dev.off()

# Print ROC for every model in the separate plots for FALSEPOSITIVE
png(filename="outputs/roc_svm_radial.comparison_FALSEPOSITIVE.png")
plot(svmRC.ROC_FALSEPOSITIVE, type = "S", col = "green", print.auc=TRUE)
garbage <- dev.off()
png(filename="outputs/roc_svm_polynomial.comparison_FALSEPOSITIVE.png")
plot(svmPC.ROC_FALSEPOSITIVE, type = "S", col = "red", print.auc=TRUE)
garbage <- dev.off()
png(filename="outputs/roc_bayes.comparison_FALSEPOSITIVE.png")
plot(bayesC.ROC_FALSEPOSITIVE, type = "S", col = "blue", print.auc=TRUE)
garbage <- dev.off()
png(filename="outputs/roc_network.comparison_FALSEPOSITIVE.png")
plot(networkC.ROC_FALSEPOSITIVE, type = "S", col = "orange", print.auc=TRUE)
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
