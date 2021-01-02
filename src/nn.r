### NEURAL NETWORK ###

# Import dependencies
library(neuralnet)
library(caret)

# Get data from generated csv
dataTrain = read.csv("datasets/tmp/train_pca.csv")
dataTrain = dataTrain[dataTrain$koi_disposition != "CANDIDATE",]
dataTest = read.csv("datasets/tmp/test_pca.csv")
dataTest = dataTest[dataTest$koi_disposition != "CANDIDATE",]

# scale
dataTrain[c(2,ncol(dataTrain))] <- scale(dataTrain[c(2,ncol(dataTrain))])
dataTest[c(2,ncol(dataTest))] <- scale(dataTest[c(2,ncol(dataTest))])
# Factorize the label
dataTrain$koi_disposition = factor(dataTrain$koi_disposition)
dataTest$koi_disposition = factor(dataTest$koi_disposition)

network = neuralnet(koi_disposition ~ ., dataTrain, hidden=c(1,1), stepmax=1e7)                       
## pesi iniziali                                                                    
msw = network$startweights                                                          
## pesi finali                                                                        
nw = network$weights                                                                
## la rete potrebbe non riuscire ad imparare                                         
steps = network$result.matrix["steps",]                                             
err = network$result.matrix["error",]                                               
                                                                                    
##plotto la rete                                                                    
plot(network)                                                                       
test.active = subset(dataTest, select=-c(koi_disposition))                          
net.predict = neuralnet::compute(network, test.active)$net.result                              
## scelgo classe con predizione più alta                                            

net.prediction = c("CONFIRMED", "FALSE POSITIVE")[apply(net.predict,    
                                                        1,                    
                                                        which.max)]    
## ne stampo la tabella di classificazione/confusione                               
predict.table = table(net.prediction, dataTest$koi_disposition)
predict.table   

result = confusionMatrix(as.factor(net.prediction), dataTest$koi_disposition, mode = "prec_recall")
result
pred.prob = attr(as.factor(net.prediction), "probabilities") 
pred.to.roc = pred.prob[, 1] 