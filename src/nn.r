### NEURAL NETWORK ###

# Import dependencies
library("neuralnet")
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
dataTrain = read.csv("datasets/tmp/train_pca.csv")
dataTrain = dataTrain[dataTrain$koi_disposition != "CANDIDATE",]
dataTest = read.csv("datasets/tmp/test_pca.csv")
dataTest = dataTest[dataTest$koi_disposition != "CANDIDATE",]

# Scale datasets
dataTrain[c(2,ncol(dataTrain))] <- scale(dataTrain[c(2,ncol(dataTrain))])
dataTest[c(2,ncol(dataTest))] <- scale(dataTest[c(2,ncol(dataTest))])

# Factorize the label
dataTrain$koi_disposition = factor(dataTrain$koi_disposition)
dataTest$koi_disposition = factor(dataTest$koi_disposition)

# Train the neural network with the neuralnet() function
# with 1 neuron in first layer and 1 neuron n second layer
network = neuralnet(koi_disposition ~ ., dataTrain, hidden=c(4,4), stepmax=1e7)

# Save generated model
saveRDS(network, "models/nn.rds")

# Compare the initial random weights with the estimated 
# weights, and visualize the overall neural network
msw = network$startweights # initial weights
nw = network$weights # final weights

# Extra info
steps = network$result.matrix["steps",]                                             
err = network$result.matrix["error",]                                               
                                                                                    
# Plot the neural network
plot(network)            

# Remove target from testset
test.active = subset(dataTest, select=-c(koi_disposition))

# Predict: use the compute function
# to obtain a probability distribution of labels 
# for each test instance
net.predict = neuralnet::compute(network, test.active)$net.result

# Insert column names in net.predict matrix
colnames(net.predict) <- c("CONFIRMED", "FALSE POSITIVE")

# To obtain the predicted class value for each test instance, take
# the class label with the highest probability
net.prediction = c("CONFIRMED", "FALSE POSITIVE")[apply(net.predict, 1, which.max)]
net.pred = predict(network, dataTest, probability = TRUE)

# Generate a classification table based on ground truth labels 
# and predicted labels, then compute accuracy
predict.table = table(net.prediction, dataTest$koi_disposition)
# and print it
predict.table 

# Create our confusion matrix
result = confusionMatrix(predict.table, mode = "prec_recall")
# and print it
result

## Calculate ROC curve
# Obtain the probability of labels with "CONFIRMED"
pred.prob = net.pred # it's already ok
pred.to.roc = pred.prob[, 1] 

# Use the performance function to obtain the performance measurement
# And use a custom label ordering 
# https://stackoverflow.com/questions/54148554/roc-curve-for-perfect-labeling-is-produced-upside-down-by-package-rocr
pred.rocr = prediction(pred.to.roc, dataTest$koi_disposition, label.ordering = c("FALSE POSITIVE", "CONFIRMED")) 
    
# Use the performance function to obtain the performance measurement
perf.rocr = performance(pred.rocr, measure = "auc", x.measure= "cutoff") 
perf.tpr.rocr = performance(pred.rocr, "tpr","fpr")

# Visualize the ROC curve using the plot function
plot(perf.tpr.rocr, colorize=T, main=paste("AUC:",(perf.rocr@y.values))) 

# Plot the random classifier
abline(a=0, b=1)

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
print(c(index = ind, accuracy = acc, cutoff = cutoff))
