library(tidyverse)
library(data.table)
library(neuralnet)
library(caret)
library(grDevices)

# HEART DISEASE PREDICTION DATA SETS ----
# Read in the training and test data sets from the working directory
traindatafile <- "./train.csv"
train <- read.csv(traindatafile, header = TRUE, sep = ",", colClasses = c(rep("integer", times = 9),"numeric",rep("integer", times = 4)))

testdatafile <- "./test.csv"
test <- read.csv(testdatafile, header = TRUE, sep = ",", colClasses = c(rep("integer", times = 9),"numeric",rep("integer", times = 4)))

# Center Training data in each column to zero mean and scale (normalize) by the standard deviation in each column.
scaledTrain <- scale(train, center = TRUE, scale = TRUE) 

# Create a Covariance Matrix of Training Data & Target
covTrain <- cov(scaledTrain) 

# Create a Heatmap to visualize the covariance matrix from the training data
heatmap(covTrain, Rowv = NA, Colv = NA, distfun = dist, hclustfun = hclust, add.expr, symm = TRUE, revC = TRUE, scale = "none", na.rm = TRUE, 
        cexRow = 0.5 + 1/log10(nrow(covTrain)), cexCol = 0.5 + 1/log10(ncol(covTrain)), labRow = rownames(covTrain), labCol = colnames(covTrain), 
        margins = c(5, 5), main = "Covariance Heatmap of Heart Disease Variables & Target Outcome", col = hcl.colors(16, palette = "Blue-Red", rev = TRUE))  
legend("left", legend = seq(from = 1, to = -0.6, by = -0.1), fill = hcl.colors(17, palette = "Blue-Red", rev = FALSE), 
       bty = "n", x.intersp = 0.1, y.intersp = 1, inset = -0.15, cex = 2)

# Drop the Target data (column 14) from the training dataset and save as XtrainScaled to indicate the scaled X training data without the target y training data
XtrainScaled <- scaledTrain[,1:13] 

# Center the TEST data in each column to zero mean and scale (normalize) by the standard deviation in each column.
scaledTest <- scale(test, center = TRUE, scale = TRUE)  

# Drop the TEST data (column 14) from the test dataset and save as XtestScaled to indicate the scaled X test data without the target y test data
XtestScaled <- scaledTest[,1:13] 

# Verify that Target Training y data is RANDOMLY ORDERED between No Heart Disease = 0 and Heart Disease = 1 
# which prevents overfitting when training neural networks.
train$target  

# Tally the train$target data with table function to see the total counts of No Heart Disease = 0 and Heart Disease = 1
# Should be a fairly balanced dataset, therefore no need to downSample Heart Disease data
table(train$target)



# 16,8,4,2 MODELS ----
# Run neuralnet with the training data on a 16,8,4,2 FNN model with the following settings, 
# which are considered to be the default settings for this project:
set.seed(1, sample.kind = "Rounding")          
nnTrained.0 <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(16,8,4,2), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE) 

# display the error, threshold reached and number of steps
as.matrix(nnTrained.0$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.0 <- predict(nnTrained.0, XtestScaled)
PredictedTest.0 <- ifelse(nnPredictTest.0 > 0.5, 1, 0)

# Display the Truth & Predicted observations table:
Xtab.0 <- table(Predicted  = PredictedTest.0, Truth = test$target)
Xtab.0 <- Xtab.0[rev(1:nrow(Xtab.0)),rev(1:ncol(Xtab.0))]
Xtab.0

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.0)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.0)



# Run another neuralnet with the training data on a 16,8,4,2 FNN model with the following settings, 
# which tests the performance of the "backprop" algorithm for comparison to the previous run with the "rprop+" algorithm:
set.seed(1, sample.kind = "Rounding")          
nnTrained.0a <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(16,8,4,2), threshold = 0.01, stepmax = 1e+06, rep = 1, 
                          learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = 0.001, lifesign = "full", lifesign.step = 1000, 
                          algorithm = "backprop", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)            

# display the error, threshold reached and number of steps
as.matrix(nnTrained.0a$result.matrix[1:3,1])                                                           

# Use the trained model to predict the test data outcomes                          
nnPredictTest.0a <- predict(nnTrained.0a, XtestScaled)
PredictedTest.0a <- ifelse(nnPredictTest.0a > 0.5, 1, 0)                                                   

# Display the Truth & Predicted observations table:                                                  
Xtab.0a <- table(Predicted  = PredictedTest.0a, Truth = test$target)
Xtab.0a <- Xtab.0a[rev(1:nrow(Xtab.0a)),rev(1:ncol(Xtab.0a))]
Xtab.0a

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.0a)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.0a)



# Run another neuralnet with the training data on a 16,8,4,2 FNN model with the following settings, 
# which tests the performance of the "rprop-" algorithm for comparison to the previous 2 runs with the other algorithms:
set.seed(1, sample.kind = "Rounding")          
nnTrained.0b <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(16,8,4,2), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                          learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                          algorithm = "rprop-", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)            

# display the error, threshold reached and number of steps
as.matrix(nnTrained.0b$result.matrix[1:3,1])                                                           
                                                           
# Use the trained model to predict the test data outcomes 
nnPredictTest.0b <- predict(nnTrained.0b, XtestScaled)
PredictedTest.0b <- ifelse(nnPredictTest.0b > 0.5, 1, 0)                                                   

# Display the Truth & Predicted observations table:        
Xtab.0b <- table(Predicted  = PredictedTest.0b, Truth = test$target)
Xtab.0b <- Xtab.0b[rev(1:nrow(Xtab.0b)),rev(1:ncol(Xtab.0b))]
Xtab.0b

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.0b)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.0b)



# 12.6.3 DEFAULT MODEL ----
# Run neuralnet with the training data on a 12,6,3 FNN model with the following settings, 
# which are considered to be the default settings for this project:
set.seed(1, sample.kind = "Rounding")          
nnTrained.1 <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(12,6,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.1$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.1 <- predict(nnTrained.1, XtestScaled)
PredictedTest.1 <- ifelse(nnPredictTest.1 > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.1 <- table(Predicted  = PredictedTest.1, Truth = test$target)
Xtab.1 <- Xtab.1[rev(1:nrow(Xtab.1)),rev(1:ncol(Xtab.1))]
Xtab.1

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.1)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.1)



# 10,5,3 DEFAULT MODEL ----
# Run neuralnet with the training data on a 10,5,3 FNN model with the following settings, 
# which are considered to be the default settings for this project:
set.seed(1, sample.kind = "Rounding")          
nnTrained.2 <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(10,5,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.2$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.2 <- predict(nnTrained.2, XtestScaled)
PredictedTest.2 <- ifelse(nnPredictTest.2 > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.2 <- table(Predicted  = PredictedTest.2, Truth = test$target)
Xtab.2 <- Xtab.2[rev(1:nrow(Xtab.2)),rev(1:ncol(Xtab.2))]
Xtab.2

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.2)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.2)



# 14,7,3 DEFAULT MODEL ----
# Run neuralnet with the training data on a 14,7,3 FNN model with the following settings, 
# which are considered to be the default settings for this project:
set.seed(1, sample.kind = "Rounding")          
nnTrained.3 <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(14,7,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.3$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.3 <- predict(nnTrained.3, XtestScaled)
PredictedTest.3 <- ifelse(nnPredictTest.3 > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.3 <- table(Predicted  = PredictedTest.3, Truth = test$target)
Xtab.3 <- Xtab.3[rev(1:nrow(Xtab.3)),rev(1:ncol(Xtab.3))]
Xtab.3

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.3)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.3)



# 13,7,3 DEFAULT MODEL ----
# Run neuralnet with the training data on a 13,7,3 FNN model with the following settings, 
# which are considered to be the default settings for this project:
set.seed(1, sample.kind = "Rounding")          
nnTrained.4 <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(13,7,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 500, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.4$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.4 <- predict(nnTrained.4, XtestScaled)
PredictedTest.4 <- ifelse(nnPredictTest.4 > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.4 <- table(Predicted  = PredictedTest.4, Truth = test$target)
Xtab.4 <- Xtab.4[rev(1:nrow(Xtab.4)),rev(1:ncol(Xtab.4))]
Xtab.4

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.4)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.4)



# 13,6,3 DEFAULT MODEL ----
# Run neuralnet with the training data on a 13,6,3 FNN model with the following settings, 
# which are considered to be the default settings for this project:
set.seed(1, sample.kind = "Rounding")          
nnTrained.5 <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(13,6,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)

# display the error, threshold reached and number of steps
as.matrix(nnTrained.5$result.matrix[1:3,1])       

# Use the trained model to predict the test data outcomes
nnPredictTest.5 <- predict(nnTrained.5, XtestScaled)
PredictedTest.5 <- ifelse(nnPredictTest.5 > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.5 <- table(Predicted  = PredictedTest.5, Truth = test$target)
Xtab.5 <- Xtab.5[rev(1:nrow(Xtab.5)),rev(1:ncol(Xtab.5))]
Xtab.5

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(t(Xtab.5))

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.5)



# CODE FOR PLOTS OF NEURAL NETWORK NODES WITH FINAL BIASES AND WEIGHTS:  ----   
# Change the first argument to match the trained model of interest:
plot(nnTrained.5, col.entry.synapse = "black", 
     col.entry = "black", col.hidden = "blue", 
     col.hidden.synapse = "blue", col.out = "black", 
     col.out.synapse = "black", col.intercept = "red", 
     fontsize = 12, dimension = 6) 



# 13,7,3 SET.SEED(4)  MODEL ----
# Run another neuralnet with the training data on the 13,7,3 FNN model with the following default settings except for set.seed(). 
# Test model performance using a set.seed(4) to control the startweights (for comparison to the previous 13,7,3 FNN model that 
# was run with set.seed at the default setting):
set.seed(4, sample.kind = "Rounding")         
nnTrained.4a <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(13,7,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                          learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 50, 
                          algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.4a$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.4a <- predict(nnTrained.4a, XtestScaled)
PredictedTest.4a <- ifelse(nnPredictTest.4a > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.4a <- table(Predicted  = PredictedTest.4a, Truth = test$target)
Xtab.4a <- Xtab.4a[rev(1:nrow(Xtab.4a)),rev(1:ncol(Xtab.4a))]
Xtab.4a

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.4a)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.4a)

 

# 10,5,3 SET.SEED(4) MODEL ----
# Run another neuralnet with the training data on the 10,5,3 FNN model with the following default settings except for set.seed(). 
# Test model performance using a set.seed(4) to control the startweights (for comparison to the previous 10,5,3 FNN model that 
# was run with set.seed at the default setting):
set.seed(4, sample.kind = "Rounding")          
nnTrained.2a <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(10,5,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                          learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 500, 
                          algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.2a$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.2a <- predict(nnTrained.2a, XtestScaled)
PredictedTest.2a <- ifelse(nnPredictTest.2a > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.2a <- table(Predicted  = PredictedTest.2a, Truth = test$target)
Xtab.2a <- Xtab.2a[rev(1:nrow(Xtab.2a)),rev(1:ncol(Xtab.2a))]
Xtab.2a

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.2a)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.2a)

# Review the startweights used in this model 
# generated by neuralnet using seed = 4 with rnorm mean = 0 and rnorm std dev = 1
nnTrained.2a$startweights  

# flatten the start weights matrix into a vector that can be plotted in a histogram
nnTrained.2aFLATstartweights <- unlist(nnTrained.2a$startweights)

# plot histogram of the 10,5,3 model start weights generated using set.seed(4) ----
hist_nnTrained.2aFLATstartweights <- hist(nnTrained.2aFLATstartweights, breaks = 50, xlim = c(-3, 3), ylim = c(0, 20),
                                          main = "10,5,3 NN Model Start Weights from set.seed(4)",
                                          xlab = "Start Weights", ylab = "Frequency")
mtext("Used for 'nnTrained.2a' NN Model Start Weights", side = 3)   # top - one line down from main
mtext("set.seed(4)                                     ", side = 3, line = -1.5)   # top - 2.5 lines down from main
mtext("rnorm(n = 217, mean = 0, sd = 1)", side = 3, line = -2.5)   # top - 3.5 lines down from main
mtext("Subfigure B", side = 1, adj = 1, line = 4)   # bottom right, 4 lines down from x-axis

# Review the startweights used in the previous 10,5,3 model with all default settings, including set.seed(1)
# generated by neuralnet using seed = 1 with rnorm mean = 0 and rnorm std dev = 1
nnTrained.2$startweights  

# flatten the start weights matrix into a vector that can be plotted in a histogram
nnTrained.2FLATstartweights <- unlist(nnTrained.2$startweights)

# plot histogram of the 10,5,3 model start weights generated using set.seed(1) ----
hist_nnTrained.2FLATstartweights <- hist(nnTrained.2FLATstartweights, breaks = 50, xlim = c(-3, 3), ylim = c(0, 20),
                                         main = "10,5,3 NN Model Start Weights from set.seed(1)",
                                         xlab = "Start Weights", ylab = "Frequency")
mtext("Used for 'nnTrained.2' NN Model Start Weights", side = 3)   # top - one line down from main
mtext("set.seed(1)                                     ", side = 3, line = -1.5)   # top - 2.5 lines down from main
mtext("rnorm(n = 217, mean = 0, sd = 1)", side = 3, line = -2.5)   # top - 3.5 lines down from main
mtext("Subfigure A", side = 1, adj = 1, line = 4)   # bottom right, 4 lines down from x-axis

# verify that rnorm(n = 217, mean = 0, sd = 1) creates a start weights vector that is identical
# to the start weights created by the neuralnet function when used with set.seed(1) for the 10,5,3 model:
set.seed(1)  
seed1_rnorm_m0sd1 <- rnorm(n = 217, mean = 0, sd = 1)
seed1_rnorm_m0sd1


# Create a conditioned set of initialized Start Weights for the 10,5,3 NN Model based on set.seed(1) with the rnorm function ----       
# Per p. 20 of LeCun et al. (1998), the Standard Deviation = 1/sqrt(number of connections feeding into a node) for rnorm of Start Weights:

1/sqrt(13)  # 13 inputs feeding into each node in 1st hidden layer
# [1] 0.2773501

1/sqrt(10)  # 10 connections feeding into each node in 2nd hidden layer
# [1] 0.3162278  # Use this value for start weights as a good compromise across the 4 levels of connections into the nodes of the 3 hidden layers 
               # and the output node.
1/sqrt(5)  # 5 connections feeding into each node in 3rd hidden layer
# [1] 0.4472136

1/sqrt(3)  # 3 connections feeding into the output node
# [1] 0.5773503

# Need 217 start weights for 10.5.3 model, and weights that are not too close to zero, i.e., absolute value of weight >= 0.0100000000
set.seed(1)
startweights10.5.3 <- rnorm(n = 217, mean = 0, sd = 1/sqrt(10))
sum(abs(startweights10.5.3)<0.0100000000)
# [1] 6  # 6 weights are too small

set.seed(1)  # add 6 to 217 = 223 and rerun
startweights10.5.3 <- rnorm(n = 223, mean = 0, sd = 1/sqrt(10))
sum(abs(startweights10.5.3)<0.0100000000)
# [1] 7  # 7 weights are too small

set.seed(1)  # add 7 to 217 = 224 and rerun
startweights10.5.3 <- rnorm(n = 224, mean = 0, sd = 1/sqrt(10))
sum(abs(startweights10.5.3)<0.0100000000)
# [1] 7  # 7 out of 224 weights are too small

# so then select for the 217 weights that are not too small:
startweights10.5.3 <- startweights10.5.3[abs(startweights10.5.3)>=0.0100000000]

# review startweights10.5.3 vector
startweights10.5.3

# verify none of the weights are too small
sum(abs(startweights10.5.3)<0.0100000000) 
# [1] 0

# plot histogram of the 10,5,3 model start weights generated using startweights10.5.3 vector ----
hist10.5.3startweights <- hist(startweights10.5.3, breaks = 32, xlim = c(-3, 3), ylim = c(0, 20),
                               main = "10,5,3 FNN Model Start Weights from startweights10.5.3",
                               xlab = "Start Weights", ylab = "Frequency") 
mtext("Used for 10,5,3 FNN Models 'nnTrained.2b' and 'nnTrained.2c'", side = 3)   # top - 1
mtext("set.seed(1)                                                                                                                  ", side = 3, line = -1.5)
mtext("rnorm(n = 224, mean = 0, sd = 1/sqrt(10))                                                              ", side = 3, line = -2.5)
mtext("7 weights too close to zero and removed from 224 weights = 217 start weights", side = 3, line = -3.5)   # top - 4.5
mtext("Subfigure C", side = 1, adj = 1, line = 4)   # bottom right, 4 lines down from x-axis



# 10,5,3 startweights10.5.3 MODEL ----
# Run another neuralnet with the training data on the 10,5,3 FNN model with the following default settings except for 
# set.seed() is omitted and startweights argument = startweights10.5.3
# to control the startweights (for comparison to the previous 10,5,3 FNN models that were run with set.seed at 1 and at 4):
nnTrained.2b <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(10,5,3), threshold = 0.01, stepmax = 1e+05, rep = 1, 
                          startweights = startweights10.5.3, learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", 
                          lifesign.step = 50, algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)

# display the error, threshold reached and number of steps
as.matrix(nnTrained.2b$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.2b <- predict(nnTrained.2b, XtestScaled)
PredictedTest.2b <- ifelse(nnPredictTest.2b > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.2b <- table(Predicted  = PredictedTest.2b, Truth = test$target)
Xtab.2b <- Xtab.2b[rev(1:nrow(Xtab.2b)),rev(1:ncol(Xtab.2b))]
Xtab.2b

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.2b)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.2b)

# Review the startweights generated by neuralnet using the startweights10.5.3 vector
# which was created (above) using seed = 1 with rnorm mean = 0 and rnorm std dev = 1/sqrt(10)
nnTrained.2b$startweights  



# 10,5,3 THRESHOLD 0.001 startweights10.5.3 MODEL ----
# Repeat the previous neuralnet run on the 10,5,3 FNN model with a lower threshold setting = 0.001
# keeping set.seed() omitted and startweights argument = startweights10.5.3
nnTrained.2c <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(10,5,3), threshold = 0.001, stepmax = 1e+05, rep = 1, 
                          startweights = startweights10.5.3, learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", 
                          lifesign.step = 50, algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)
                                                          
# display the error, threshold reached and number of steps
as.matrix(nnTrained.2c$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.2c <- predict(nnTrained.2c, XtestScaled)
PredictedTest.2c <- ifelse(nnPredictTest.2c > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.2c <- table(Predicted  = PredictedTest.2c, Truth = test$target)
Xtab.2c <- Xtab.2c[rev(1:nrow(Xtab.2c)),rev(1:ncol(Xtab.2c))]
Xtab.2c

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.2c)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.2c)

# Review the startweights generated by neuralnet using the startweights10.5.3 vector
# which was created (above) using seed = 1 with rnorm mean = 0 and rnorm std dev = 1/sqrt(10)
nnTrained.2c$weights



# 10,5,3 THRESHOLD 0.001 SET.SEED(1) MODEL ----
# Repeat the nnTrained.2 neuralnet run with the training data on a 10,5,3 FNN model with the following settings, 
# set.seed(1) and Threshold setting = 0.001
set.seed(1, sample.kind = "Rounding")          
nnTrained.2h <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(10,5,3), threshold = 0.001, stepmax = 1e+05, rep = 1, 
                         learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 500, 
                         algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.2h$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.2h <- predict(nnTrained.2h, XtestScaled)
PredictedTest.2h <- ifelse(nnPredictTest.2h > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.2h <- table(Predicted  = PredictedTest.2h, Truth = test$target)
Xtab.2h <- Xtab.2h[rev(1:nrow(Xtab.2h)),rev(1:ncol(Xtab.2h))]
Xtab.2h

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.2h)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.2h)



# 10,5,3 THRESHOLD 0.001 SET.SEED(4) MODEL ----
# Repeat the nnTrained.2a neuralnet run with the training data on a 10,5,3 FNN model with the following settings, 
# set.seed(4) and Threshold setting = 0.001
set.seed(4, sample.kind = "Rounding")          
nnTrained.2i <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(10,5,3), threshold = 0.001, stepmax = 1e+05, rep = 1, 
                          learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "full", lifesign.step = 1000, 
                          algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)    

# display the error, threshold reached and number of steps
as.matrix(nnTrained.2i$result.matrix[1:3,1])

# Use the trained model to predict the test data outcomes
nnPredictTest.2i <- predict(nnTrained.2i, XtestScaled)
PredictedTest.2i <- ifelse(nnPredictTest.2i > 0.5, 1, 0)

# Display the Truth & Predicted observations table: 
Xtab.2i <- table(Predicted  = PredictedTest.2i, Truth = test$target)
Xtab.2i <- Xtab.2i[rev(1:nrow(Xtab.2i)),rev(1:ncol(Xtab.2i))]
Xtab.2i

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.2i)

# Calculate the Confusion Matrix Results:
confusionMatrix(Xtab.2i)



# Logistic Regression Benchmark of Heart Disease Prediction Data ----

library(tidyverse)
library(data.table)
library(caret)
library(grDevices)

# convert the XtrainScaled matrix into a data table              
XtrainScaledDT <- as.data.table(XtrainScaled)

# fit a logistic regression model to the XtrainScaled data
fit_logistic <- glm(train$target ~ ., data = XtrainScaledDT, family = "binomial")

# convert the XtestScaled matrix into a data table 
XtestScaledDT <- as.data.table(XtestScaled)

# predict the XtestScaled data outcomes usinng the fit_logistic model
p_hat_logistic <- predict(fit_logistic, XtestScaledDT, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.5, 1, 0))

# Calculate the Confusion Matrix Results and Display the Truth & Predicted observations table: 
Xtab.log <- confusionMatrix(y_hat_logistic, as.factor(test$target), positive = "1")
Xtab.log

# Calculate the Matthews Correlation Coefficient (MCC)
yardstick::mcc(Xtab.log$table)



# Code for creating the Loss Landscape Plots ----
# Load the neuralnet package
library(neuralnet)

# Define the loss function
loss <- function(y_true, y_pred) {
  return(mean((y_true - y_pred)^2))
}

# Designate the split train and test sets
x_train <- XtrainScaled
y_train <- train$target
x_test <- XtestScaled
y_test <- test$target

# Choose one of the following options to Define Start Weights
# startweights <- startweights10.5.3
startweights <- NULL
set.seed(1)
# set.seed(4)
# startweights <- rnorm(n = length(nn$result.matrix)-3, mean = 0, sd = 1)

# Train a neural network with RPROP+ algorithm
nn <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(16,8,4,2), threshold = 0.01, stepmax = 1e+06, rep = 1, 
                startweights = startweights, learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
                algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE,
                lifesign = "full", lifesign.step = 1000)

# Predict on the test set
y_pre_pred <- predict(nn, x_test)
y_pred <- ifelse(y_pre_pred > 0.5, 1, 0)

# Calculate the test loss
test_loss <- loss(y_test, y_pred)

# Define a function to get the loss value at a given point in the weight parameter space
get_loss <- function(w) {
  # Set the weights of the neural network to w
  nn$weights <- w
  # Predict on the test set
  pre_pred <- predict(nn, x_test)
  pred <- ifelse(pre_pred > 0.5, 1, 0)
  # Return the loss value
  return(loss(y_test, pred))
}

# Define a function to generate a random direction vector in the weight parameter space
get_random_dir <- function() {
  # Get the number of parameters
  n_params <- length(nn$result.matrix)-3
  # Generate a random vector of the same length
  dir <- rnorm(n_params)
  # Normalize the vector
  dir <- dir / sqrt(sum(dir^2))
  # Return the direction
  return(dir)
}

# Define a function to plot the loss along a given direction
plot_loss <- function(dir, x_range) {
  # Get the threshold weights from the trained neuralnet
  w_thresh <- nn$weights
  # Get the loss value at the training threshold point
  loss_thresh <- get_loss(w_thresh)
  # Initialize a vector to store the loss values along the direction
  loss_dir <- numeric(length(x_range))
  # Loop over the x values
  for (i in 1:length(x_range)) {
    # Get the x value
    x <- x_range[i]
    # Get the point along the direction
    xdir <- lapply(dir[[1]], function(d) x * d)
    xdir <- list(xdir)
    # Element-wise addition of two nested lists of matrices
    add_nested_matrices <- function(A, B) {
      lapply(seq_along(A), function(i) mapply("+", A[[i]], B[[i]]))
    }
    w <- add_nested_matrices(w_thresh, xdir)
    # Get the loss value at that point
    loss_dir[i] <- get_loss(w)
  }
  # Plot the loss values against the x values
  plot(x_range, loss_dir, type = "l", xlab = "Proportion & Direction of Path away from Trained Weights", ylab = "Loss",
       main = "Loss Paths from Start Weights to Trained Weights")
  # Add a vertical line at the threshold point
  abline(v = 0, lty = 2)
  # Add a horizontal line at the threshold loss value
  abline(h = loss_thresh, lty = 2)
  # Add a point at the threshold point
  points(0, loss_thresh, pch = 19)
  # Choose the appropriate details to add to the plot about model type, algorithm, start weights, and threshold
  # mtext("16,8,4,2 FNN Model  nnTrained.0a                               Backprop Algorithm", side = 3, line = 0.2)
  # mtext("Start = set.seed(1) rnorm(n=409,mean=0,sd=1)                 0.01 Threshold", side = 3, line = -1.2)
  # mtext("Subfigure A", side = 1, adj = 1, line = 4)
  # mtext("16,8,4,2 FNN Model  nnTrained.0b                               RPROP-  Algorithm", side = 3, line = 0.2)
  # mtext("Start = set.seed(1) rnorm(n=409,mean=0,sd=1)                 0.01 Threshold", side = 3, line = -1.2)
  # mtext("Subfigure B", side = 1, adj = 1, line = 4)
  mtext("16,8,4,2 FNN Model  nnTrained.0                                 RPROP+ Algorithm", side = 3, line = 0.2)
  mtext("Start = set.seed(1) rnorm(n=409,mean=0,sd=1)                 0.01 Threshold", side = 3, line = -1.2)
  mtext("Subfigure C", side = 1, adj = 1, line = 4)
  # mtext("10,5,3 FNN Model  nnTrained.2c                       RPROP+ Algorithm", side = 3, line = 0.2)
  # mtext("Start = startweights10.5.3                                      0.001 Threshold", side = 3, line = -1.2)
  # mtext("Subfigure C", side = 1, adj = 1, line = 4)
  # mtext("10,5,3 FNN Model  nnTrained.2h                       RPROP+ Algorithm", side = 3, line = 0.2)
  # mtext("Start = set.seed(1) rnorm(n=217,mean=0,sd=1)     0.001 Threshold", side = 3, line = -1.2)
  # mtext("Subfigure A", side = 1, adj = 1, line = 4)
  # mtext("10,5,3 FNN Model  nnTrained.2i                         RPROP+ Algorithm", side = 3, line = 0.2)
  # mtext("Start = set.seed(4) rnorm(n=217,mean=0,sd=1)     0.001 Threshold", side = 3, line = -1.2)
  # mtext("Subfigure B", side = 1, adj = 1, line = 4)
}

# Choose the same options as code line 580 (above) to Define Start Weights 
# in order to Generate the Direction vector for plotting:
# dir <- get_random_dir()
# dir <- startweights10.5.3
dir <- NULL
set.seed(1)
# set.seed(4)
# dir <- rnorm(n = length(nn$result.matrix)-3, mean = 0, sd = 1)
# Use neuralnet to reshape dir into a list of weight matricies in the nn format, through the 'startweights' argument
dirNN <- neuralnet(train$target ~ ., data = XtrainScaled, hidden = c(16,8,4,2), threshold = 0.01, stepmax = 1e+06, rep = 1, 
                   startweights = dir, learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
                   algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE, 
                   lifesign = "full", lifesign.step = 1000)
dir <- dirNN$startweights
# dir <- nnTrained.2$startweights

# Define the x range for plotting
x_range <- seq(-1, 1, length.out = 1000)

# Plot the loss along the direction
plot_loss(dir, x_range)




