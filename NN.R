rm(list=ls())
install.packages("ISLR")
library(ISLR)

data("Auto")

X<-subset(Auto,select=c(horsepower, weight, year, origin))
Y<-subset(Auto,select=c(mpg))

#adding dummy variables for origin=1,2
X[,5]<-0
X[,6]<-0
for(i in 1:nrow(X)){
  if(X$origin[i]==1){
    X[i,5]<-1
    X[i,6]<-0
  }else if (X$origin[i]==2){
    X[i,5]<-0
    X[i,6]<-1
  }
}

as.numeric(c(5,6),data.frame(X))

#normalizing input data
standard.X <- apply(X, 2, normalize)

#dividing data into training and testing dataset
set.seed(1204)
train          = sample(1:nrow(standard.X),0.5*nrow(standard.X))
training_data  = standard.X[train,]
Y_train        = as.matrix(Y[train,])
test_data      = standard.X[-train,]
Y_test         = as.matrix(Y[-train,])

J=c()
epochs=500
a=0.01
alpha_orig=alphaFunc(training_data)
beta_orig=betaFunc()

alpha=alpha_orig
beta=beta_orig

for(i in 1:epochs){
  Z2=alpha[1,] + training_data%*%alpha[-1,]
  a2=as.matrix(sigmoid(Z2))
  y_train=beta[1,]+a2%*%beta[-1,]
  m1=DJDWalpha(beta,training_data,a2,y_train)
  m2=DJDWbeta(a2,y_train)
  #change value of alpha and beta with respect to each iteration 
  alpha[-1,]=alpha[-1,]-a*m1
  beta[-1,]=beta[-1,]-a*m2
  J=append(J,MSE(Y_train,y_train,training_data))
}

#calculate MSE for diff weights
for(i in 1:100){
  epochs=1000
  a=0.01
  alpha=alphaFunc(training_data)
  beta=betaFunc()
  Mean_Square_Error=neuralNet(alpha,beta,a,epochs,stop=TRUE)
  J=append(J,Mean_Square_Error[length(Mean_Square_Error)])
}

#calculate MSE for single weights
J=c()
epochs=500
a=0.01
alpha=alphaFunc(training_data)
beta=betaFunc()
J=neuralNet(alpha,beta,a,epochs)

plot(1:length(J),J,col='red')

#apply alpha beta after training from neural network now in test data
Z2=alpha[1,] + test_data%*%alpha[-1,]
a2=as.matrix(sigmoid(Z2))
y_test=beta[1,]+a2%*%beta[-1,]
MSE_test=MSE(Y_test,y_test,test_data)
MSE_test

#calculate MSE using linear regression model
X_LM<-subset(Auto,select=c(mpg,horsepower, weight, year, origin))
lm_mpg<-lm(mpg~.,data.frame(X_LM[train,]))
y_lm=predict(lm_mpg,data=test_data)
MSE_test_lm=MSE(Y_test,y_lm,test_data)

#comparing train MSE with and without Regularization
J=c()
J_regularization=c()
J=neuralNet(alpha,beta,a,epochs,stop=FALSE)
J_regularization=neuralNet_Regularization(alpha,beta,a,epochs,stop=FALSE)
plot(1:length(J),J,col='red')
lines(1:length(J_regularization),J_regularization,col='blue')

#training alpha and beta for train data using regularization
alpha=alpha_orig
beta=beta_orig
for(i in 1:epochs){
  J=c()
  Z2=alpha[1,] + training_data%*%alpha[-1,]
  a2=as.matrix(sigmoid(Z2))
  y_train=beta[1,]+a2%*%beta[-1,]
  m1=DJDWalpha(beta,training_data,a2,y_train)
  m2=DJDWbeta(a2,y_train)
  #change value of alpha and beta with respect to each iteration 
  alpha[-1,]=alpha[-1,]-a*m1
  beta[-1,]=beta[-1,]-a*m2
  J=append(J,MSE_Regularization(Y_train,y_train,training_data,alpha,beta))
}

#checking test MSE using Regularization using trained alpha and beta from above
Z2=alpha[1,] + test_data%*%alpha[-1,]
a2=as.matrix(sigmoid(Z2))
y_test=beta[1,]+a2%*%beta[-1,]
MSE_test_regularization=MSE(Y_test,y_test,test_data)

#MSE function using regularization
MSE_Regularization<-function(Y,y,data,alpha,beta){
  J=(0.5*sum((Y-y)^2)/nrow(data))+0.1*(sum(alpha)+sum(beta%*%t(beta)))
  return (J)
}

#normalizing input data
normalize <- function(X){
  return((X-mean(X))/sd(X))
}

#stop iterations when value of MSE doesn't change much
stoppingFunction<-function(J){
  change=((J[length(J)]-J[length(J)-1])/(J[length(J)-1]))
  if(abs(change) < 0.001){
    return (TRUE)
  }
  else{
    return (FALSE)
  }
}

#derivative of MSE with respect to beta
DJDWbeta<-function(a2,y_train){
  delta=2*(Y_train-y_train)
  djdw2=(t(a2)%*%as.matrix(delta))/nrow(training_data)
  return (-djdw2)
}

#derivative of MSE with respect to alpha
DJDWalpha<-function(beta,data,a2,y_train){
  s=(2*as.matrix(Y_train-y_train)%*%t(beta[-1,]))*(a2)
  djdw1=(t(data)%*%s)/nrow(data)
  return (-djdw1)
}

#calculation of Mean Square Error
MSE<-function(Y,y,data){
  J=0.5*sum((Y-y)^2)/nrow(data)
  return (J)
}

derivativeOfSigmoid<-function(x){
  return (exp(-x)/((1+exp(-x))^2))
}

sigmoid<-function(x){
  return (1/(1+exp(-x)))
}

#calculate alpha
alphaFunc<-function(Z){
  p = dim(Z)[2]
  m = 5
  W=matrix(runif((p+1)*m,-0.7,0.7),p+1,m)
  return (W)
}

#calculate beta
betaFunc<-function(){
  m = 5
  W=matrix(runif(m+1,-0.7,0.7),m+1,1)
  return (W)
}

neuralNet<-function(alpha,beta,a,n,stop=FALSE){
  J1=c()
  counter=0
  for(i in 1:n){
    Z2=alpha[1,] + training_data%*%alpha[-1,]
    a2=as.matrix(sigmoid(Z2))
    y_train=beta[1,]+a2%*%beta[-1,]
    m1=DJDWalpha(beta,training_data,a2,y_train)
    m2=DJDWbeta(a2,y_train)
    #change value of alpha and beta with respect to each iteration 
    alpha[-1,]=alpha[-1,]-a*m1
    beta[-1,]=beta[-1,]-a*m2
    J1=append(J1,MSE(Y_train,y_train,training_data))
    #stop iterations when value of MSE doesn't change much in last 10 values 
    if((i>10)&&(stop==TRUE)){
      if(stoppingFunction(J1)){counter=counter+1}
      else{counter=0}
      if(counter>10){break}
    }
  }
  return (J1)
}

neuralNet_Regularization<-function(alpha,beta,a,n,stop=FALSE){
  J1=c()
  counter=0
  for(i in 1:n){
    Z2=alpha[1,] + training_data%*%alpha[-1,]
    a2=as.matrix(sigmoid(Z2))
    y_train=beta[1,]+a2%*%beta[-1,]
    m1=DJDWalpha(beta,training_data,a2,y_train)
    m2=DJDWbeta(a2,y_train)
    #change value of alpha and beta with respect to each iteration 
    alpha[-1,]=alpha[-1,]-a*m1
    beta[-1,]=beta[-1,]-a*m2
    J1=append(J1,MSE_Regularization(Y_train,y_train,training_data,alpha,beta))
    #stop iterations when value of MSE doesn't change much in last 10 values 
    if((i>10)&&(stop==TRUE)){
      if(stoppingFunction(J1)){counter=counter+1}
      else{counter=0}
      if(counter>10){break}
    }
  }
  return (J1)
}
