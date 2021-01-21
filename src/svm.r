### SVM ###

# Import dependencies
library("e1071")
library("caret")
library("ROCR") 
library("dplyr") 

# Function to determinate the optimal cutoff
opt.cut= function(perf, pred){
    cut.ind= mapply(FUN=function(x, y, p){
        d = (x -0)^2 + (y-1)^2
        ind = which(d == min(d))
        c(sensitivity= y[[ind]], specificity= 1-x[[ind]], cutoff= p[[ind]])
        }, perf@x.values, perf@y.values, pred@cutoffs)
    }

# Get data from generated csv
dataTrain = read.csv("datasets/out/train_pca.csv")
dataTrain = dataTrain[dataTrain$koi_disposition != "CANDIDATE",]
dataTest = read.csv("datasets/out/test_pca.csv")
dataTest = dataTest[dataTest$koi_disposition != "CANDIDATE",]

# Scale datasets
dataTrain[c(2, ncol(dataTrain))] <- scale(dataTrain[c(2,ncol(dataTrain))])
dataTest[c(2, ncol(dataTest))] <- scale(dataTest[c(2,ncol(dataTest))])

# Factorize the label
dataTrain$koi_disposition = factor(dataTrain$koi_disposition)
dataTest$koi_disposition = factor(dataTest$koi_disposition)

## RADIAL

# Train SVM 
# based on dataTrain
svm.modelR = svm(koi_disposition ~ .,
                 data = dataTrain,
                 kernel = 'radial',
                 prob=TRUE)

# Save generated model
saveRDS(svm.modelR, "models/svm.radial.rds")

# Predict the label of new instances (of dataTest) 
# by using the already trained SVM model
svm.predR = predict(svm.modelR, dataTest, probability=TRUE)

# Create our confusion matrix
svm.tableR = table(svm.predR, dataTest$koi_disposition)
resultR = confusionMatrix(svm.predR, dataTest$koi_disposition, mode = "prec_recall")
# And print results and confusion matrix
svm.tableR
resultR

## Calculate ROC curve
# Obtain the probability of labels with "CONFIRMED"
pred.probR = attr(svm.predR, "probabilities") 
pred.to.rocR = pred.probR[, 1] 

# Use the performance function to obtain the performance measurement
# And use a custom label ordering 
# https://stackoverflow.com/questions/54148554/roc-curve-for-perfect-labeling-is-produced-upside-down-by-package-rocr
pred.rocrR = prediction(pred.to.rocR, dataTest$koi_disposition, label.ordering = c("FALSE POSITIVE", "CONFIRMED")) 

# Use the performance function to obtain the performance measurement
perf.rocrR = performance(pred.rocrR, measure = "auc", x.measure= "cutoff") 
perf.tpr.rocrR = performance(pred.rocrR, "tpr","fpr")

# Visualize the ROC curve using the plot function
plot(perf.tpr.rocrR, colorize=T,main=paste("AUC:",(perf.rocrR@y.values))) 

# Plot the random classifier
abline(a=0, b=1)

# Print optimal cutoff
print("cutoff")
print(opt.cut(perf.tpr.rocrR, pred.rocrR))

# Get the overall accuracy for the simple predictions 
acc.perfR = performance(pred.rocrR, measure= "acc")
# and plot it
plot(acc.perfR)

# Grab the index for maximum accuracy and then grab the corresponding cutoff
indR = which.max(slot(acc.perfR, "y.values")[[1]])
accR = slot(acc.perfR, "y.values")[[1]][indR]
cutoffR = slot(acc.perfR, "x.values")[[1]][indR]
# And print results
print(c(index = indR, accuracy = accR, cutoff = cutoffR))


## POLYNOMIAL

# Train SVM 
# based on dataTrain
svm.modelP = svm(koi_disposition ~ .,
                 data = dataTrain,
                 kernel = 'polynomial',
                 grades = c(2,3,4,5,6,7,8,9,10,11),
                 coef0 = c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                 prob = TRUE)

# Save generated model
saveRDS(svm.modelP, "models/svm.polynomial.rds")

# Predict the label of new instances (of dataTest) 
# by using the already trained SVM model
svm.predP = predict(svm.modelP,
                    dataTest,
                    probability=TRUE)

# Create our confusion matrix
svm.tableP = table(svm.predP, dataTest$koi_disposition)
resultP = confusionMatrix(svm.predP, dataTest$koi_disposition, mode = "prec_recall")
# And print results and confusion matrix
svm.tableP
resultP

## Calculate ROC curve
# Obtain the probability of labels with "CONFIRMED"
pred.probP = attr(svm.predP, "probabilities") 
pred.to.rocP = pred.probP[, 1] 

# Use the performance function to obtain the performance measurement
# And use a custom label ordering 
# https://stackoverflow.com/questions/54148554/roc-curve-for-perfect-labeling-is-produced-upside-down-by-package-rocr
pred.rocrP = prediction(pred.to.rocP, dataTest$koi_disposition, label.ordering = c("FALSE POSITIVE", "CONFIRMED")) 

# Use the performance function to obtain the performance measurement
perf.rocrP = performance(pred.rocrP, measure = "auc", x.measure= "cutoff") 
perf.tpr.rocrP = performance(pred.rocrP, "tpr","fpr")

# Visualize the ROC curve using the plot function
plot(perf.tpr.rocrP, colorize=T,main=paste("AUC:",(perf.rocrP@y.values))) 

# Plot the random classifier
abline(a=0,b=1)

# Print optimal cutoff
print("cutoff")
print(opt.cut(perf.tpr.rocrP, pred.rocrP))

# Get the overall accuracy for the simple predictions 
acc.perfP = performance(pred.rocrP, measure= "acc")
# and plot it
plot(acc.perfP)

# Grab the index for maximum accuracy and then grab the corresponding cutoff
indP = which.max(slot(acc.perfP, "y.values")[[1]])
accP = slot(acc.perfP, "y.values")[[1]][indP]
cutoffP = slot(acc.perfP, "x.values")[[1]][indP]
# And print results
print(c(index = indP, accuracy = accP, cutoff = cutoffP))
