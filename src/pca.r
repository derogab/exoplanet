### PCA ###

# Import dependencies
library("dplyr")
library("FactoMineR")
library("factoextra")
library("corrplot")
library("groupdata2")

# Get data from csv
data <- read.csv('datasets/cumulative_2021.01.15_07.47.15.csv', header = TRUE) 

# Remove rows w/ target koi_vet_stat = active
data <- data[data$koi_vet_stat != "Active",]

# Remove rows w/ target koi_disposition = candidate or not dispositioned
data <- data[data$koi_disposition != "CANDIDATE",]
data <- data[data$koi_disposition != "NOT DISPOSITIONED",]

# Remove useless and not numeric column
data <- subset(data, 
               select=-c(rowid, 
                         kepid, 
                         kepoi_name, 
                         kepler_name,
                         koi_vet_date,                  
                         koi_pdisposition,
                         koi_score,
                         koi_fpflag_nt,
                         koi_fpflag_ss,
                         koi_fpflag_co,
                         koi_fpflag_ec,
                         koi_disp_prov,
                         koi_comment,
                         koi_eccen_err1,
                         koi_eccen_err2,
                         koi_longp,
                         koi_longp_err1,
                         koi_longp_err2,
                         koi_ingress,
                         koi_ingress_err1,
                         koi_ingress_err2,
                         koi_sma_err1,
                         koi_sma_err2,
                         koi_incl_err1,
                         koi_incl_err2,
                         koi_tce_delivname, 
                         koi_teq_err1, 
                         koi_teq_err2,
                         koi_model_dof,
                         koi_model_chisq,
                         koi_datalink_dvr, 
                         koi_datalink_dvs,
                         koi_sage,
                         koi_sage_err1,
                         koi_sage_err2,
                         koi_fittype,
                         koi_limbdark_mod,
                         koi_trans_mod,
                         koi_sparprov,
                         koi_vet_stat,
                         koi_parm_prov))

# Label as factor
data$koi_disposition <- as.factor(data$koi_disposition)

# Clear row with n/a
data <- data[complete.cases(data),]

# Plot distribution of target before downsample
plot(data$koi_disposition)

# Downsample the data based on target (koi_disposition)
data <- downsample(data, cat_col="koi_disposition")

# Plot distribution of target after downsample 
plot(data$koi_disposition)

# Remove label from train set
data.active <- subset(data, select=-c(koi_disposition))

# Correlation matrix
M<-cor(data.active, use="pairwise.complete.obs")
head(round(M,2))
corrplot(M)

# Execute PCA
res.pca <- PCA(data.active, 
               scale.unit=TRUE, 
               graph=FALSE, 
               ncp=ncol(data.active))

# Extract eigenvalues and graph
eig.val <- get_eigenvalue(res.pca)
eig.graph = fviz_eig(res.pca, 
                     addlabels=TRUE, 
                     ncp=ncol(data.active))
                     

# Get PCA variance
var <- get_pca_var(res.pca)

# Corrplot
# The contributions of variables in accounting for the variability in a 
# given principal component are expressed in percentage
corrplot(cor(var$contrib))   

# Count number of dimensions with eigenvalue >= 1
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

# Get PCA train and test
train.pca = select(train, attributes)
test.pca = select(test, attributes)

# Write csv files
write.csv(train, "datasets/tmp/train.csv", row.names = FALSE)
write.csv(test, "datasets/tmp/test.csv", row.names = FALSE)
write.csv(train.pca, "datasets/tmp/train_pca.csv", row.names = FALSE)
write.csv(test.pca, "datasets/tmp/test_pca.csv", row.names = FALSE)
