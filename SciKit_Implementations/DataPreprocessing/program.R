#importing the dataset
dataset=read.csv('Data.csv')
#changing the missing values to mean of column
dataset$Age=ifelse(is.na(dataset$Age),ave(dataset$Age, FUN =function(x) mean(x, na.rm=TRUE))
                   ,dataset$Age)


dataset$Salary=ifelse(is.na(dataset$Salary),ave(dataset$Salary, FUN =function(x) mean(x, na.rm=TRUE))
                   ,dataset$Salary)
#Encoding Categorical data
dataset$Country=factor(dataset$Country,levels=c('France', 'Germany', 'Spain'),labels=c(1,2,3))
dataset$Purchased=factor(dataset$Purchased,levels=c('No','Yes'),labels=c(0,1))

#Now splitting the data
#install.packages('caTools')
library(caTools)
split=sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
                      
#feature scaling Age and Salary
training_set[ ,2:3] = scale(training_set[ ,2:3])
test_set[, 2:3]= scale(test_set[, 2:3])
