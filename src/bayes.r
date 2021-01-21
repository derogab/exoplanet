### NAIVE BAYES ###

# Import dependencies
library("e1071")
library("caret")
library("ROCR")

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
levels(dataTrain$koi_disposition) <- make.names(levels(factor(dataTrain$koi_disposition)))
levels(dataTest$koi_disposition) <- make.names(levels(factor(dataTest$koi_disposition)))

# Create bayes classifier
bayes_classifier = naiveBayes(dataTrain, dataTrain$koi_disposition, laplace = c(0, 1, 2, 3))

# Save generated model
saveRDS(bayes_classifier, "models/bayes.rds")

# and train the model
bayes.pred.raw = predict(bayes_classifier, dataTest, type = "raw")
bayes.pred = predict(bayes_classifier, dataTest) 

# Create our confusion matrix
bayes.table = table(bayes.pred, dataTest$koi_disposition)
result = confusionMatrix(bayes.pred, dataTest$koi_disposition, mode = "prec_recall")
# And print results and confusion matrix
bayes.table
result

## Calculate ROC curve
# Obtain the probability of labels with "CONFIRMED"
pred.prob = bayes.pred.raw # it's already ok
pred.to.roc = pred.prob[, 1] 

# Use the performance function to obtain the performance measurement
# And use a custom label ordering 
# https://stackoverflow.com/questions/54148554/roc-curve-for-perfect-labeling-is-produced-upside-down-by-package-rocr
pred.rocr = prediction(pred.to.roc, dataTest$koi_disposition,
                       label.ordering = c("FALSE.POSITIVE", "CONFIRMED")) 

# Use the performance function to obtain the performance measurement
perf.rocr = performance(pred.rocr, measure = "auc", x.measure= "cutoff") 
perf.tpr.rocr = performance(pred.rocr, "tpr", "fpr")

# Visualize the ROC curve using the plot function
plot(perf.tpr.rocr, colorize=T, main=paste("AUC:",(perf.rocr@y.values))) 

# Plot the random classifier
abline(a=0,b=1)

# Print optimal cutoff
print("cutoff")
print(opt.cut(perf.tpr.rocr, pred.rocr))

# Get the overall accuracy for the simple predictions 
acc.perf = performance(pred.rocr, measure= "acc")
# and plot it
plot(acc.perf)

# Grab the index for maximum accuracy and then grab the corresponding cutoff
ind = which.max(slot(acc.perf, "y.values")[[1]])
acc = slot(acc.perf, "y.values")[[1]][ind]
cutoff = slot(acc.perf, "x.values")[[1]][ind]
# And print results
print(c(accuracy = acc, cutoff = cutoff))
