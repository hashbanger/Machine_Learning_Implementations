#importing the dataset
dataset = read.csv('Data.csv')
#dataset = dataset[ ,2:3]
#Now splitting the data
#install.packages('caTools')
library(caTools)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


#Now feature scaling 
# training_set[ ,2:3] = scale(training_set[ ,2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
