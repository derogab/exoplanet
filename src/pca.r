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
                         koi_eccen, ## always 0
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
                         koi_parm_prov,
                         koi_period_err2,
                         koi_time0bk_err2,
                         koi_time0_err2,
                         koi_duration_err2,
                         koi_depth_err2,
                         koi_dor_err2))

# Label as factor
data$koi_disposition <- as.factor(data$koi_disposition)

# Clear row with n/a
data <- data[complete.cases(data),]
cat("Number of features: ", ncol(data),"\n")
# Plot distribution of target before downsample
png(filename="outputs/before_downsample_distr.pca.png")
plot(data$koi_disposition)
garbage <- dev.off()

# Downsample the data based on target (koi_disposition)
data <- downsample(data, cat_col="koi_disposition")

# Plot distribution of target after downsample 
png(filename="outputs/after_downsample_distr.pca.png")
plot(data$koi_disposition)
garbage <- dev.off()

# Remove label from train set
data.active <- subset(data, select=-c(koi_disposition))

# Correlation matrix
#M<-cor(data.active, use="pairwise.complete.obs")
#head(round(M,2))
#png(filename="outputs/corr_matrix.pca.png")
#corrplot(M)
#dev.off()

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
png(filename="outputs/distr.pca.png", width = 1480, height = 550)                 
eig.graph
garbage <- dev.off()
#png(filename="outputs/pca_distr_test.png", width = 1480, height = 550)
#fviz_pca_var(res.pca, col.var = "black")
#dev.off()

#png(filename="outputs/pca_distr_lol.png", width = 1480, height = 550)                 
#fviz_eig(res.pca,
#         addlabels = T, 
#         barcolor = "#E7B800", 
#         barfill = "#E7B800", 
#         linecolor = "#00AFBB", 
#         choice = "variance", 
#         ylim=c(0,10))
#dev.off()


# Get PCA variance
var <- get_pca_var(res.pca)

# Corrplot
# The contributions of variables in accounting for the variability in a 
# given principal component are expressed in percentage
#png(filename="outputs/corr_matrix_after_pca.png")
#corrplot(cor(var$contrib), is.corr=FALSE)   
#dev.off()

# Count number of dimensions with eigenvalue > 1
count_eig <- 0
for(i in 1:nrow(eig.val)){
  if(eig.val[i] > 1){
    count_eig <- count_eig + 1
  }
}

# Print the component with eig.val > 1
cat('Count of component with eig.val > 1: ', count_eig, '\n')

# Print the cumulative variance at last component with eig.val > 1
cat('Cumulative variance at last component with eig.val > 1: ', eig.val[count_eig, 3], '%\n')

# Count the number of dimensions to reach ~70% variance
seventy = 0
for (x in 1:length(eig.val[,3])){
    if(eig.val[x,3] >= 70){
        seventy = x-1
        break
    }
}
png(filename="outputs/cumulative_variance.pca.png")
plot(eig.val[,3], type="l", xlab="Dimensions", ylab="Cumulative Variance")
# Red line for dims with eigenvalues > 1
abline(v=count_eig, col=2, lty="dashed")
# Green line for 70% variance
abline(v=seventy, col=3, lty="dashed")

garbage <- dev.off()


# Contributions of variables to principal components
png(filename="outputs/hist_pca.png", width = 1480, height = 550)
fviz_contrib(res.pca, choice = "var", axes = 1:count_eig, title = "")
garbage <- dev.off()
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
png(filename="outputs/train_distr.pca.png")
plot(train$koi_disposition)
garbage <- dev.off()
png(filename="outputs/test_distr.pca.png")
plot(test$koi_disposition)
garbage <- dev.off()

# Get PCA train and test
train.pca = select(train, all_of(attributes))
test.pca = select(test, all_of(attributes))

# Write csv files
write.csv(train, "datasets/tmp/train.csv", row.names = FALSE)
write.csv(test, "datasets/tmp/test.csv", row.names = FALSE)
write.csv(train.pca, "datasets/tmp/train_pca.csv", row.names = FALSE)
write.csv(test.pca, "datasets/tmp/test_pca.csv", row.names = FALSE)
