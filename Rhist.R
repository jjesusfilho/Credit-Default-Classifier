summary(ticdata2000)
train_data = ticdata2000
train_data = data.frame(lapply(train_data, as.factor))
summary(train_data)
myData = train_data


myData[,81:85] = NULL
myData[,77:79] = NULL
myData[,69:75] = NULL
myData[,66:67] = NULL
myData[,60:64] = NULL
myData[,48:58] = NULL
myData[,45:46] = NULL

test_data = ticeval2000
#test_data = data.frame(lapply(test_data, as.factor))
myData2 = test_data
dim(test_data)

dim(myData2)

myData2[,81:85] = NULL
myData2[,77:79] = NULL
myData2[,69:75] = NULL
myData2[,66:67] = NULL
myData2[,60:64] = NULL
myData2[,48:58] = NULL
myData2[,45:46] = NULL

lr.model = glm(X86~., data=myData, family="binomial")
summary(lr.model)
#not a great reduction in residual deviance


lr.model$xlevels[["X3"]] <- union(lr.model$xlevels[["X3"]], levels(myData2$X3))
lr.model$xlevels[["X29"]] <- union(lr.model$xlevels[["X29"]], levels(myData2$X29))
lr.model$xlevels[["X33"]] <- union(lr.model$xlevels[["X33"]], levels(myData2$X33))
lr.model$xlevels[["X41"]] <- union(lr.model$xlevels[["X41"]], levels(myData2$X41))
lr.model$xlevels[["X47"]] <- union(lr.model$xlevels[["X47"]], levels(myData2$X47))
lr.model$xlevels[["X68"]] <- union(lr.model$xlevels[["X68"]], levels(myData2$X68))
lr.model$xlevels[["X76"]] <- union(lr.model$xlevels[["X76"]], levels(myData2$X76))
lr.model$xlevels[["X80"]] <- union(lr.model$xlevels[["X80"]], levels(myData2$X80))


lr.pred = predict(lr.model,newdata = myData2, type="response")
pred = rep(0,4000)
pred[lr.pred > .5] = 1
table(myData2$X86,pred)


for(i in 1:50)
{
  levels(myData2[,i]) = union(levels(myData[,i]), levels(myData2[,i]))
}

library("randomForest")
bag.bands = randomForest(X86~., data=myData, mtry=6,importance=TRUE)
bag.bands.pred = predict(bag.bands,myData2,type="class")
table(bag.bands.pred,test_labels$X1)
dim(myData2)


myData1 = scale(myData)
#nn_Data = model.matrix(~., data=myData)[,-1]
#nn_Data = as.data.frame(nn_Data)
#n = names(nn_Data)
n1 = names(myData)
f = as.formula(paste("X86 ~", paste(n1[!n1 %in% "X86"], collapse = " + ")))
f
nn.fit = neuralnet(f,data=myData1,hidden=c(3,2,3),linear.output=FALSE, threshold = .01)


myData3 = scale(myData2)
pr.nn = compute(nn.fit,myData3)
nn_bandPred = pr.nn$net.result
nn_pred = rep(0,4000)
nn_pred[nn_bandPred > .4] = 1
table(nn_pred,test_labels$X1)
mean(nn_pred == test_labels$X1)

?neuralnet

myData=as.data.frame(myData)
myData2=as.data.frame(myData2)
train.x = data.matrix(myData[, 1:49])
train.y = myData[, 50]
test.x = data.matrix(myData2[, 1:49])
test.y = myData2[, 50]

maxs = apply(train.x, 2, max)
mins = apply(train.x, 2, min)
# These are vectors of the max and min values for each colum
train.x = as.data.frame(scale(train.x, center = mins, scale = maxs - mins))

maxs = apply(test.x, 2, max)
mins = apply(test.x, 2, min)
# These are vectors of the max and min values for each colum
train.x = as.data.frame(scale(test.x, center = mins, scale = maxs - mins))

model <- mx.mlp(train.x, train.y, hidden_node=c(6,3), out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9,
                eval.metric=mx.metric.accuracy)
preds = predict(model, test.x)
pred.label = max.col(t(preds))-1
table(pred.label, test.y)
mean(pred.label == test.y)

#GLM


#Regression MOdel
myData=as.data.frame(unclass(AmesHousing))

myData = myData[myData$Gr.Liv.Area < 4000,]

myData$Overall.Cond = as.factor(myData$Overall.Cond)
myData$Overall.Qual = as.factor(myData$Overall.Qual)

myData = myData[,-1]
myData = myData[,-1]

myData = myData[,c("MS.SubClass","MS.Zoning", "Lot.Frontage","Lot.Area","Lot.Shape"
                   ,"Land.Contour","Lot.Config","Neighborhood","Bldg.Type","House.Style",
                   "Overall.Qual","Overall.Cond","Roof.Style",
                   "BsmtFin.Type.1","Bsmt.Unf.SF","X1st.Flr.SF","Gr.Liv.Area",
                   "Full.Bath","TotRms.AbvGrd","SalePrice")]
myData =  na.omit(myData)

train = sample(2369,1500)
train.x = data.matrix(myData[train, -20])
train.y = myData[train, 20]
test.x = data.matrix(myData[-train, -20])
test.y = myData[-train, 20]

data <- mx.symbol.Variable("myData")
# A fully connected hidden layer
# data: input source
# num_hidden: number of neurons in this hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=6)
# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc1)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(),     num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9,  eval.metric=mx.metric.rmse)
preds = predict(model, test.x)
sqrt(mean((preds-test.y)^2))



?gbm
library(gbm)
boost.bank = gbm(X86~., data=myData, distribution="adaboost", n.trees=50, shrinkage=.01)
boost.bank
boost.probs = predict(boost.bank, newdata=myData2, n.trees=50, type="response")
head(boost.probs)
dim(myData2)
boost.pred = rep(0,4000)
boost.pred[boost.probs>.065]=1
table(boost.pred,test_labels$X1)
best.iter=gbm.perf(boost.bank,method = "OOB")
print(best.iter)
dim(myData)
