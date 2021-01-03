### PCA ###

# Import dependencies
library("dplyr")
library("FactoMineR")
library("factoextra")
library("corrplot")

# Get data from csv
data <- read.csv('datasets/raw_data.csv', header = TRUE)

# Remove useless and not numeric column
data <- subset(data, 
               select=-c(rowid, 
                         kepid, 
                         kepoi_name, 
                         kepler_name,
                         koi_fpflag_nt,
                         koi_fpflag_ss,
                         koi_fpflag_co,
                         koi_fpflag_ec,
                         koi_pdisposition, 
                         koi_tce_delivname, 
                         koi_teq_err1, 
                         koi_teq_err2,
                         koi_score))

# Clear row with n/a
data <- data[complete.cases(data),]

# Factorize the label
data$koi_disposition <- as.factor(data$koi_disposition)

# Plot distribution of target
plot(data$koi_disposition)

# Split function
split.data = function(data, p = 0.7, s = 1){
    set.seed(s)
    index = sample(1:dim(data)[1])
    train = data[index[1:floor(dim(data)[1] * p)], ]
    test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
    return(list(train=train, test=test))
}

# Split the dataset to train + test
all = split.data(data, p=0.7) # split 70% - 30%
train = all$train # get train set
test = all$test # get test set

# Plot train and test distribution of target
plot(train$koi_disposition)
plot(test$koi_disposition)

# Remove label from train set
train.active <- subset(train, select=-c(koi_disposition))

# Execute PCA
res.pca <- PCA(train.active, 
               scale.unit=TRUE, 
               graph=FALSE, 
               ncp=ncol(train.active))

# Extract eigenvalues and graph
eig.val <- get_eigenvalue(res.pca)
eig.graph = fviz_eig(res.pca, 
                     addlabels=TRUE, 
                     ncp=ncol(train.active))
                     

# Get PCA variance
var <- get_pca_var(res.pca)

# Corrplot
# The contributions of variables in accounting for the variability in a 
# given principal component are expressed in percentage
corrplot(var$contrib, is.corr=FALSE)   

# Count number of dimentions with eigenvalue >= 1
count_eig <- 0
for(i in 1:nrow(eig.val)){
  if(eig.val[i]>= 1){
    count_eig <- count_eig + 1
  }
}

# Contributions of variables to principal components
fviz_contrib(res.pca, choice = "var", axes = 1:count_eig, top = 40)

# Get first count_eig contributions of individuals
var_contrib = head(var$contrib, count_eig)

# Get the names of significative attributes
attributes <- rownames(var_contrib)

# Add target column to found columns
attributes <- c("koi_disposition", attributes)

# Get PCA train and test
train.pca = select(train, attributes)
test.pca = select(test, attributes)

# Write csv files
write.csv(train, "datasets/tmp/trainnb.csv", row.names = FALSE)
write.csv(test, "datasets/tmp/testnb.csv", row.names = FALSE)
write.csv(train.pca, "datasets/tmp/train_pcanb.csv", row.names = FALSE)
write.csv(test.pca, "datasets/tmp/test_pcanb.csv", row.names = FALSE)
