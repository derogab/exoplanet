### NEURAL NETWORK ###

# Import dependencies
library(neuralnet)
library(caret)

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
network = neuralnet(koi_disposition ~ ., dataTrain, hidden=c(1,1), stepmax=1e7)                       

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

# To obtain the predicted class value for each test instance, take
# the class label with the highest probability
net.prediction = c("CONFIRMED", "FALSE POSITIVE")[apply(net.predict, 1, which.max)]

# Generate a classification table based on ground truth labels 
# and predicted labels, then compute accuracy
predict.table = table(net.prediction, dataTest$koi_disposition)
# and print it
predict.table 

# Create our confusion matrix
result = confusionMatrix(as.factor(net.prediction), dataTest$koi_disposition, mode = "prec_recall")
# and print it
result

## Calculate ROC curve
# Obtain the probability of labels with "CONFIRMED"
pred.prob = attr(as.factor(net.prediction), "probabilities") 
pred.to.roc = pred.prob[, 1] 
