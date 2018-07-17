rm(list=ls())

#########################################################################
################ -- Machine Learning for Data Science -- ################
#########################################################################

# ######### 1) LIBRERIAS A UTILIZAR ################# 

library(sqldf)
library(ggvis)
library(party)
library(Boruta)
library(pROC)
library(randomForest)
library(e1071)
library(caret)
library(glmnet)
library(mboost)
library(adabag)
library(xgboost)
library(ROCR)
library(C50)
library(mlr)
library(lattice)
library(gmodels)
library(gplots)
library(DMwR)
library(rminer)
library(polycor)
library(class)
library(neuralnet)

######### 2) EXTRAYENDO LA DATA ################# 

train<-read.csv("train.csv",na.strings = c(""," ",NA)) # leer la data de entrenamiento
test<-read.csv("test.csv",na.strings = c(""," ",NA))  # leer la data de Validacion 

names(train) # visualizar los nombres de la data
head(train)  # visualizar los 6 primeros registros
str(train)   # ver la estructura de la data

######### 3) EXPLORACION DE LA DATA ################# 

# tablas resumen
summary(train) # tabla comun de obtener
summarizeColumns(train) # tabla mas completa

resumen=data.frame(summarizeColumns(train))

## Graficos para variables cuantitativas

# histogramas y Cajas

#Veamos la variable ApplicantIncome
hist(train$ApplicantIncome, breaks = 100, main = "Applicant Income Chart",xlab = "ApplicantIncome",col="blue")

#Veamos la variable CoapplicantIncome
hist(train$CoapplicantIncome, breaks = 100, main = "Coapplicant Income Chart",xlab = "CoapplicantIncome",col="red")

#Veamos los Outliers
#Se visualizan valores atipicos

#Veamos la variable LoanAmount
bwplot(train$LoanAmount, layout = c(1, 1),main = "Loan Amount Chart",xlab = "LoanAmount", col="blue")

#Veamos la variable Loan_Amount_Term
bwplot(train$Loan_Amount_Term, layout = c(1, 1),main = "Loan Amount Term Chart",xlab = "Loan_Amount_Term", col="blue")

## Graficos para variables cuantitativas

#Veamos la variable Gender
CrossTable(train$Gender,prop.t=FALSE,prop.r=TRUE,prop.c=FALSE,prop.chisq=FALSE)

#Veamos la variable Married
CrossTable(train$Married,prop.t=FALSE,prop.r=TRUE,prop.c=FALSE,prop.chisq=FALSE)

#Veamos la variable Loan_Status
CrossTable(train$Loan_Status,prop.t=FALSE,prop.r=TRUE,prop.c=FALSE,prop.chisq=FALSE)

# Graficos de cualitativos
Tabla1=table(train$Gender)
Tabla2=table(train$Married)

par(mfrow=c(1,2))
balloonplot(t(Tabla1), main ="Tabla de Contingencia Gender",xlab ="Gender", label = FALSE, show.margins = FALSE)
balloonplot(t(Tabla2), main ="Tabla de Contingencia Married",xlab ="Married", label = FALSE, show.margins = FALSE)

# comentarios de la data

# 1. LoanAmount tiene (614 - 592) 22 valores perdidos.
# 2. Loan_Amount_Term tiene (614 - 600) 14 valores perdidos.
# 3. Credit_History tiene (614 - 564) 50 valores perdidos.
# 4. Nosotros podemos tambi�n observar que cerca del 84% de los solicitantes al pr�stamo 
# tienen un historial crediticio. �C�mo? La media del campo Credit_History es 0.84 
# (Recordemos, Credit_History tiene o toma el valor 1 para aquellos que tienen 
#   historial crediticio y 0 en caso contrario).
# 5. La variable ApplicantIncome parece estar en l�nea con las espectativas al 
# igual que CoapplicantIncome.

######### 4) IMPUTACION DE LA DATA ################# 

# revisar valores perdidos

perdidos=data.frame(resumen$name,resumen$na,resumen$type); colnames(perdidos)=c("name","na","type")
perdidos

# recodificando Dependents
train$Dependents=ifelse(train$Dependents=="3+",3,
                                 ifelse(train$Dependents=="0",0,
                                        ifelse(train$Dependents=="1",1,
                                               ifelse(train$Dependents=="2",2,
                                                      train$Dependents))))
train$Dependents=as.factor(train$Dependents)

# convirtiendo en factor Credit_History
train$Credit_History <- as.factor(train$Credit_History)

# recodificando Dependents
test$Dependents=ifelse(test$Dependents=="3+",3,
                        ifelse(test$Dependents=="0",0,
                               ifelse(test$Dependents=="1",1,
                                      ifelse(test$Dependents=="2",2,
                                             test$Dependents))))
test$Dependents=as.factor(test$Dependents)

# convirtiendo en factor Credit_History
test$Credit_History <- as.factor(test$Credit_History)

# recodificando Loan_Status
train$Loan_Status=ifelse(train$Loan_Status=="N",0,1)
train$Loan_Status=as.factor(train$Loan_Status)

# partcionando la data en numericos y factores

numericos <- sapply(train, is.numeric) # variables cuantitativas
factores <- sapply(train, is.factor)  # variables cualitativas

train_numericos <-  train[ , numericos]
train_factores <- train[ , factores]

# APLICAR LA FUNCION LAPPLY PARA DISTINTAS COLUMNAS CONVERTIR A FORMATO NUMERICO
n1=min(dim(train_factores))
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.numeric)
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.factor)

numericos <- sapply(test, is.numeric) # variables cuantitativas
factores <- sapply(test, is.factor)  # variables cualitativas

test_numericos <-  test[ , numericos]
test_factores <- test[ , factores]

# APLICAR LA FUNCION LAPPLY PARA DISTINTAS COLUMNAS CONVERTIR A FORMATO NUMERICO
n1=min(dim(test_factores))
test_factores[2:(n1)] <- lapply(test_factores[2:(n1)], as.numeric)
test_factores[2:(n1)] <- lapply(test_factores[2:(n1)], as.factor)

# Para train y test

train=cbind(train_numericos,train_factores[,-1])
test=cbind(test_numericos,test_factores[,-1])

## Imputacion Parametrica

#Podemos imputar los valores perdidos por la media o la moda

# data train
train_parametrica <- impute(train, classes = list(factor = imputeMode(), 
                                    integer = imputeMean()),
              dummy.classes = c("integer","factor"), dummy.type = "numeric")
train_parametrica=train_parametrica$data[,1:min(dim(train))]

# data test
test_parametrica  <- impute(test, classes = list(factor = imputeMode(), 
                                    integer = imputeMean()), 
               dummy.classes = c("integer","factor"), dummy.type = "numeric")
test_parametrica=test_parametrica$data[,1:min(dim(test))]

summary(train_parametrica)

# Imputacion No parametrica

#Imputacion con algoritmos de ML

# random forest
train_no_parametrica <- rfImpute(Loan_Status~.,train)
summary(train_no_parametrica)

# MEDIANTE KNN 
train_no_parametrica2=knnImputation(train, k = 10, scale = T)
summary(train_no_parametrica2)

######### 5) CREACION  Y TRANSFORMACION DE VARIABLES ################# 

train_parametrica$Total_income=train_parametrica$ApplicantIncome+train_parametrica$CoapplicantIncome
train_parametrica$log_LoanAmount=train_parametrica$LoanAmount
train_parametrica$Amauntxterm=train_parametrica$Total_income/train_parametrica$LoanAmount

calcula_indicadores <- function(objeto_logit)
{
  objeto_logit.ks <- ks.test(x = objeto_logit$fitted.values[which(objeto_logit$y == 0)],
                             y = objeto_logit$fitted.values[which(objeto_logit$y == 1)])
  objeto_logit.ks$statistic
  
  objeto_logit.roc <- roc(response=objeto_logit$y,predictor=objeto_logit$fitted.values)
  objeto_logit.gini <- 2*objeto_logit.roc$auc-1
  
  return(unname(c(objeto_logit.gini,objeto_logit.ks$statistic)))
}


pdf("recodificacion.pdf")

## ApplicantIncome
datos.tree<-ctree(Loan_Status ~ ApplicantIncome
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("ApplicantIncome (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## CoapplicantIncome
datos.tree<-ctree(Loan_Status ~ CoapplicantIncome
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("CoapplicantIncome (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## LoanAmount
datos.tree<-ctree(Loan_Status ~ LoanAmount
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("LoanAmount (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## Loan_Amount_Term
datos.tree<-ctree(Loan_Status ~ Loan_Amount_Term
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Loan_Amount_Term (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## Gender
datos.tree<-ctree(Loan_Status ~ Gender
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Gender (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## Married 
datos.tree<-ctree(Loan_Status ~ Married 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Married (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## Dependents 
datos.tree<-ctree(Loan_Status ~ Dependents 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Dependents (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## Education 
datos.tree<-ctree(Loan_Status ~ Education 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Education (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

## Self_Employed 
datos.tree<-ctree(Loan_Status ~ Self_Employed 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))
#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Self_Employed (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

## Credit_History 
datos.tree<-ctree(Loan_Status ~ Credit_History 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))

#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Credit_History (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

## Property_Area 

datos.tree<-ctree(Loan_Status ~ Property_Area 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))

#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Property_Area (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

# variables creadas

## Total_income 

datos.tree<-ctree(Loan_Status ~ Total_income 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))

#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Total_income (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

## log_LoanAmount 

datos.tree<-ctree(Loan_Status ~ log_LoanAmount 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))

#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("log_LoanAmount (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

## Amauntxterm 

datos.tree<-ctree(Loan_Status ~ Amauntxterm 
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))

#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("Amauntxterm (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")


## mix Edu _Marri

datos.tree<-ctree(Loan_Status ~ Education + Married
                  ,data=train_parametrica, 
                  controls=ctree_control(mincriterion=0.95))

#Estimacion
n1=dim(train_parametrica)
yprob=sapply(predict(datos.tree, newdata=train_parametrica,type="prob"),'[[',2)
fit.roc1<-roc(train_parametrica$Loan_Status, yprob)
#Gini
datos.tree.gini = 2*c(fit.roc1$auc)-1
#Indicador Kolmogorov-Smirnov
datos.tree.ks = ks.test(yprob[train_parametrica$Loan_Status==1],yprob[train_parametrica$Loan_Status==0])
plot(datos.tree,main=paste("mix Edu _Marri (GINI:", round(100*datos.tree.gini,2),"% | KS:", round(100*datos.tree.ks$statistic,2), "%)","Casos: ", n1[1],  sep=" "), cex=0.5,type="simple")

dev.off()

# Recodificacion de las variables finales

train_parametrica=sqldf("select Loan_Status,ApplicantIncome,CoapplicantIncome,LoanAmount,
                    Loan_Amount_Term,Gender,Dependents,Self_Employed,
                    Credit_History,Total_income,log_LoanAmount,Amauntxterm,
                    case 
                    when Property_Area in (1,3) then 1 else 2 end Property_Area,
                    case 
                    when Education in (2) then 1 
                    when Married in (1) then 2 
                    when Education in (1) then 3 end Edu_Ma
                    from train_parametrica")

######### 6) BALANCEO DE LOS DATOS Y SELECCION DE DRIVERS #################

## Particonando la Data

set.seed(1234)
sample <- createDataPartition(train_parametrica$Loan_Status, p = .70,
                                  list = FALSE,
                                  times = 1)

data.train <- train_parametrica[ sample,]
data.prueba <- train_parametrica[-sample,]

# Balanceo de los datos

# Balanceo mediante SMOTE

data_smoote <- SMOTE(Loan_Status ~ .,data.train  , perc.over = 100, perc.under=200)
table(data_smoote$Loan_Status)

# Nota SMOTE:
# Vamos a crear observaciones positivas adicionales usando SMOTE.
# Establecimos perc.over = 100 duplicar la cantidad de casos positivos 
# y configuramos perc.under=200 para mantener la mitad de lo que se cre� 
# como casos negativos.

# Balanceo mediante UnderSampling

menorcero<-subset(data.train,Loan_Status=="0") 
mayoruno<-subset(data.train,Loan_Status=="1")

set.seed(1234)
sample<-sample.int(nrow(mayoruno),nrow(menorcero))
length(sample)
mayoruno.prueba<-mayoruno[sample,]

data.train=rbind(mayoruno.prueba,menorcero)
table(data.train$Loan_Status)

rm(mayoruno,menorcero,mayoruno.prueba)

## seleccion de variables

##  Mediante ML

# Utilizando Boruta

pdf("seleccion de variables.pdf")
Boruta(Loan_Status~.,data=data.train,doTrace=2)->Bor.hvo;
plot(Bor.hvo,las=3);

# Utilizando RF

set.seed(1234)
rand <- randomForest( Loan_Status ~ ., data = data.train,   # Datos a entrenar 
                      ntree=100,           # N�mero de �rboles
                      mtry = 3,            # Cantidad de variables
                      importance = TRUE,   # Determina la importancia de las variables
                      replace=T)           # muestras con reemplazo

varImpPlot(rand)

# Utilizando Naive Bayes

naive <- fit(Loan_Status~., data=data.train, model="naiveBayes")
naive.imp <- Importance(naive, data=data.train)
impor.naive=data.frame(naive.imp$imp); rownames(impor.naive)=colnames(data.train)
barplot(naive.imp$imp,horiz = FALSE,names.arg = colnames(data.train),las=2)

dev.off()

## Mediante Modelos Parametricos

n1=min(dim(train_factores))
train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.numeric)
x=data.train
x[2:min(dim(data.train))]=lapply(data.train[2:min(dim(data.train))],as.numeric)
x=x[2:min(dim(data.train))]

predictores.train=as.matrix(x)
y.train=as.factor(data.train$Loan_Status)

foundlasso<-cv.glmnet(predictores.train,y.train,alpha=1,family="binomial",type.measure = "class")
par(mfrow = c(1,1))
plot(foundlasso)

foundlasso$lambda.1se
foundlasso$lambda.min
coef1=coef(foundlasso,s=foundlasso$lambda.1se)
coef1
coef2=coef(foundlasso,s=foundlasso$lambda.min)
coef2

# matriz de correlaciones no parametricas completas
correlaciones=hetcor(data.train, use = "pairwise.complete.obs")
correlaciones
correlaciones=correlaciones$correlations

# guardamos las correlaciones
#write.csv(correlaciones,"correlaciones.csv")

# Grafico del comportamiento de las variables
data.train %>% ggvis(~Total_income, ~LoanAmount, fill = ~Loan_Status) %>% layer_points()


######### 7) MODELADO DE LA DATA #################

# data de entrenamiento
data.train.1=subset(data.train,select=c("Credit_History","LoanAmount","Total_income","Amauntxterm" ,"Loan_Status"))

# data de validacion
data.test.1=subset(data.prueba,select=c("Credit_History","LoanAmount","Total_income","Amauntxterm" ,"Loan_Status"))

m=min(dim(data.train.1))


# modelo 1.- Logistico

modelo1=glm(Loan_Status~.,data=data.train.1,family = binomial(link = "logit"))
summary(modelo1)

proba1=predict(modelo1, newdata=data.test.1,type="response")

AUC1 <- roc(data.test.1$Loan_Status, proba1)

## calcular el AUC
auc_modelo1=AUC1$auc

## calcular el GINI
gini1 <- 2*(AUC1$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo1,data.test.1,type="response")
PRED=ifelse(PRED<=0.5,0,1)
PRED=as.factor(PRED)

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity1=as.numeric(tabla$byClass[1])

# Precision
Accuracy1=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error1=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo1
gini1
Accuracy1
error1
Sensitivity1

# modelo 2.- KNN

# Utilizar libreria para ML, tratamiento de la data

Credit_History=ifelse(data.train.1$Credit_History=="1",1,2)
data.train.2=data.frame(Credit_History,scale(data.train.1[2:4]),data.train.1$Loan_Status)
colnames(data.train.2)=names(data.train.1)

Credit_History=ifelse(data.test.1$Credit_History=="1",1,2)
data.test.2=data.frame(Credit_History,scale(data.test.1[2:4]),data.test.1$Loan_Status)
colnames(data.test.2)=names(data.test.1)

#create a task
trainTask <- makeClassifTask(data = data.train.2,target = "Loan_Status")
testTask <- makeClassifTask(data = data.test.2, target = "Loan_Status")

trainTask <- makeClassifTask(data = data.train.2,target = "Loan_Status", positive = "1")

# Modelado KNN

set.seed(1234)
knn <- makeLearner("classif.knn",prob = TRUE,k = 1)

qmodel <- train(knn, trainTask)
qpredict <- predict(qmodel, testTask)

response=as.numeric(qpredict$data$response[1:183])
response=ifelse(response==2,1,0)
proba2=response

# curva ROC
AUC2 <- roc(data.test.1$Loan_Status, proba2) 
auc_modelo2=AUC2$auc

# Gini
gini2 <- 2*(AUC2$auc) -1

# Calcular los valores predichos
PRED <-response

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity2=as.numeric(tabla$byClass[1])

# Precision
Accuracy2=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error2=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo2
gini2
Accuracy2
error2
Sensitivity2

# modelo 3.- Naive Bayes

modelo3=naiveBayes(Loan_Status~.,data = data.train.1)

##probabilidades
proba3<-predict(modelo3, newdata=data.test.1,type="raw")
proba3=proba3[,2]

# curva ROC
AUC3 <- roc(data.test.1$Loan_Status, proba3) 
auc_modelo3=AUC3$auc

# Gini
gini3 <- 2*(AUC3$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo3,data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity3=as.numeric(tabla$byClass[1])

# Precision
Accuracy3=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error3=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo3
gini3
Accuracy3
error3
Sensitivity3

# modelo 4.- Arbol CHAID

modelo4<-ctree(Loan_Status~.,data = data.train.1, 
               controls=ctree_control(mincriterion=0.95))

##probabilidades
proba4=sapply(predict(modelo4, newdata=data.test.1,type="prob"),'[[',2)

# curva ROC	
AUC4 <- roc(data.test.1$Loan_Status, proba4) 
auc_modelo4=AUC4$auc

# Gini
gini4 <- 2*(AUC4$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo4, newdata=data.test.1,type="response")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity4=as.numeric(tabla$byClass[1])

# Precision
Accuracy4=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error4=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo4
gini4
Accuracy4
error4
Sensitivity4

# modelo 5.- Arbol CART 

arbol.completo <- rpart(Loan_Status~.,data = data.train.1,method="class",cp=0, minbucket=0)
xerr <- arbol.completo$cptable[,"xerror"] ## error de la validacion cruzada
minxerr <- which.min(xerr)
mincp <- arbol.completo$cptable[minxerr, "CP"]

modelo5 <- prune(arbol.completo,cp=mincp)

##probabilidades
proba5=predict(modelo5, newdata=data.test.1,type="prob")[,2]

# curva ROC
AUC5 <- roc(data.test.1$Loan_Status, proba5) 
auc_modelo5=AUC5$auc

# Gini
gini5 <- 2*(AUC5$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo5, newdata=data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity5=as.numeric(tabla$byClass[1])

# Precision
Accuracy5=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error5=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo5
gini5
Accuracy5
error5
Sensitivity5

# modelo 6.- Arbol c5.0

modelo6 <- C5.0(Loan_Status~.,data = data.train.1,trials = 55,rules= TRUE,tree=FALSE,winnow=FALSE)

##probabilidades
proba6=predict(modelo6, newdata=data.test.1,type="prob")[,2]

# curva ROC
AUC6 <- roc(data.test.1$Loan_Status, proba6) 
auc_modelo6=AUC6$auc

# Gini
gini6 <- 2*(AUC6$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo6, newdata=data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity6=as.numeric(tabla$byClass[1])

# Precision
Accuracy6=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error6=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo6
gini6
Accuracy6
error6
Sensitivity6

# modelo 7.- SVM Radial

modelo7=svm(Loan_Status~.,data = data.train.1,kernel="radial",costo=100,gamma=1,probability = TRUE, method="C-classification")

##probabilidades
proba7<-predict(modelo7, newdata=data.test.1,decision.values = TRUE, probability = TRUE) 
proba7=attributes(proba7)$probabilities[,2]

# curva ROC
AUC7 <- roc(data.test.1$Loan_Status, proba7) 
auc_modelo7=AUC7$auc

# Gini
gini7 <- 2*(AUC7$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo7,data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity7=as.numeric(tabla$byClass[1])

# Precision
Accuracy7=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error7=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo7
gini7
Accuracy7
error7
Sensitivity7

# modelo 8.- SVM Linear

modelo8=svm(Loan_Status~.,data = data.train.1,kernel="linear",costo=100,probability = TRUE, method="C-classification")

##probabilidades
proba8<-predict(modelo8, newdata=data.test.1,decision.values = TRUE, probability = TRUE) 
proba8=attributes(proba8)$probabilities[,2]

# curva ROC
AUC8 <- roc(data.test.1$Loan_Status, proba8) 
auc_modelo8=AUC8$auc

# Gini
gini8 <- 2*(AUC8$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo8,data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity8=as.numeric(tabla$byClass[1])

# Precision
Accuracy8=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error8=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo8
gini8
Accuracy8
error8
Sensitivity8

# modelo 9.- SVM sigmoid

modelo9=svm(Loan_Status~.,data = data.train.1,kernel="sigmoid",costo=100,probability = TRUE, method="C-classification")

##probabilidades
proba9<-predict(modelo9, newdata=data.test.1,decision.values = TRUE, probability = TRUE) 
proba9=attributes(proba9)$probabilities[,2]

# curva ROC
AUC9 <- roc(data.test.1$Loan_Status, proba9) 
auc_modelo9=AUC9$auc

# Gini
gini9 <- 2*(AUC9$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo9,data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity9=as.numeric(tabla$byClass[1])

# Precision
Accuracy9=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error9=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo9
gini9
Accuracy9
error9
Sensitivity9

# modelo 10.- Random Forest

set.seed(1234)
modelo10 <- randomForest( Loan_Status~.,data = data.train.1,   # Datos a entrenar 
                         ntree=100,           # N�mero de �rboles
                         mtry = 1,            # Cantidad de variables
                         importance = TRUE,   # Determina la importancia de las variables
                         replace=T) 

##probabilidades
proba10<-predict(modelo10, newdata=data.test.1,type="prob")
proba10=proba10[,2]

# curva ROC
AUC10 <- roc(data.test.1$Loan_Status, proba10) 
auc_modelo10=AUC10$auc

# Gini
gini10 <- 2*(AUC10$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo10,data.test.1,type="class")

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity10=as.numeric(tabla$byClass[1])

# Precision
Accuracy10=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error10=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo10
gini10
Accuracy10
error10
Sensitivity10

# modelo 11.- Redes Neuronales

set.seed(1234)
neuralnet.learner <- makeLearner("classif.neuralnet",predict.type = "prob",hidden=c(10,15),
                                 act.fct = "logistic",algorithm = "rprop+",threshold = 0.01,stepmax = 2e+05)

qmodel <- train(neuralnet.learner, trainTask)
qpredict <- predict(qmodel, testTask)

##probabilidades
proba11=qpredict$data$prob.1

# curva ROC
AUC11 <- roc(data.test.1$Loan_Status, proba11) 
auc_modelo11=AUC11$auc

# Gini
gini11 <- 2*(AUC11$auc) -1

# Calcular los valores predichos
PRED <-qpredict$data$response

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity11=as.numeric(tabla$byClass[1])

# Precision
Accuracy11=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error11=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo11
gini11
Accuracy11
error11
Sensitivity11


# modelo 12.- Boosting

set.seed(1234)
modelo12<-boosting(Loan_Status~.,data = data.train.1)

##probabilidades
proba<-predict(modelo12,data.test.1)
proba12=(proba)$prob[,2]

# curva ROC
AUC12 <- roc(data.test.1$Loan_Status, proba12) 
auc_modelo12=AUC12$auc

# Gini
gini12 <- 2*(AUC12$auc) -1

# Calcular los valores predichos
PRED <-proba$class

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity12=as.numeric(tabla$byClass[1])

# Precision
Accuracy12=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error12=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo12
gini12
Accuracy12
error12
Sensitivity12

rm(data_smoote,data.test.2,correlaciones,data.train.2,predictores.train,train,train_no_parametrica,train_no_parametrica2,x)

# modelo 13.- XGboosting

set.seed(1234)
getParamSet("classif.xgboost")

xg_set <- makeLearner("classif.xgboost",objective = "binary:logistic", predict.type = "prob",
                      max_depth=3,eta=0.5,nthread=2,nrounds=4)

qmodel <- train(xg_set, trainTask)
qpredict <- predict(qmodel, testTask)

##probabilidades
proba13=qpredict$data$prob.1

# curva ROC
AUC13 <- roc(data.test.1$Loan_Status, proba13) 
auc_modelo13=AUC13$auc

# Gini
gini13 <- 2*(AUC13$auc) -1

# Calcular los valores predichos
PRED <-qpredict$data$response

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity13=as.numeric(tabla$byClass[1])

# Precision
Accuracy13=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error13=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo13
gini13
Accuracy13
error13
Sensitivity13

# modelo 14.- Ensamble de Modelos (CART, CHAID, C5.0)

##probabilidades

ensamble=data.frame(proba4,proba5,proba6);colnames(ensamble)=c("CHAID","CART","C50")
ensamble$ensamble=apply(ensamble, 1, mean)
ensamble$response=ifelse(ensamble$ensamble<=0.5,0,1)

proba14=ensamble$ensamble

# curva ROC
AUC14 <- roc(data.test.1$Loan_Status, proba14) 
auc_modelo14=AUC14$auc

# Gini
gini14 <- 2*(AUC14$auc) -1

# Calcular los valores predichos
PRED <-ensamble$response

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity14=as.numeric(tabla$byClass[1])

# Precision
Accuracy14=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error14=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo14
gini14
Accuracy14
error14
Sensitivity14

# modelo 15.- Stacking de Modelos (Redes Neuronales, RF, XGBoosting)

stacking=data.frame(proba10,proba11,proba13);colnames(stacking)=c("RN","RF","XGB")
stacking$stacking=apply(stacking, 1, mean)
stacking$response=ifelse(stacking$stacking<=0.5,0,1)

proba15=stacking$stacking

# curva ROC
AUC15 <- roc(data.test.1$Loan_Status, proba15) 
auc_modelo15=AUC15$auc

# Gini
gini15 <- 2*(AUC15$auc) -1

# Calcular los valores predichos
PRED <-stacking$response

# Calcular la matriz de confusi�n
tabla=confusionMatrix(PRED,data.test.1$Loan_Status,positive = "1")

# sensibilidad
Sensitivity15=as.numeric(tabla$byClass[1])

# Precision
Accuracy15=tabla$overall[1]

# Calcular el error de mala clasificaci�n
error15=mean(PRED!=data.test.1$Loan_Status)

# indicadores
auc_modelo15
gini15
Accuracy15
error15
Sensitivity15

## --Tabla De Resultados ####

AUC=rbind(auc_modelo1,
          auc_modelo2,
          auc_modelo3,
          auc_modelo4,
          auc_modelo5,
          auc_modelo6,
          auc_modelo7,
          auc_modelo8,
          auc_modelo9,
          auc_modelo10,
          auc_modelo11,
          auc_modelo12,
          auc_modelo13,
          auc_modelo14,
          auc_modelo15)
GINI=rbind(gini1,
           gini2,
           gini3,
           gini4,
           gini5,
           gini6,
           gini7,
           gini8,
           gini9,
           gini10,
           gini11,
           gini12,
           gini13,
           gini14,
           gini15)
Accuracy=rbind(Accuracy1,
            Accuracy2,
            Accuracy3,
            Accuracy4,
            Accuracy5,
            Accuracy6,
            Accuracy7,
            Accuracy8,
            Accuracy9,
            Accuracy10,
            Accuracy11,
            Accuracy12,
            Accuracy13,
            Accuracy14,
            Accuracy15
)

ERROR= rbind(error1,
             error2,
             error3,
             error4,
             error5,
             error6,
             error7,
             error8,
             error9,
             error10,
             error11,
             error12,
             error13,
             error14,
             error15
)
SENSIBILIDAD=rbind(Sensitivity1,
                   Sensitivity2,
                   Sensitivity3,
                   Sensitivity4,
                   Sensitivity5,
                   Sensitivity6,
                   Sensitivity7,
                   Sensitivity8,
                   Sensitivity9,
                   Sensitivity10,
                   Sensitivity11,
                   Sensitivity12,
                   Sensitivity13,
                   Sensitivity14,
                   Sensitivity15
)

resultado=data.frame(AUC,GINI,Accuracy,ERROR,SENSIBILIDAD)
rownames(resultado)=c('Logistico',
                      'KNN',
                      'Naive_Bayes',
                      'Arbol_CHAID',
                      'Arbol_CART ',
                      'Arbol_c50',
                      'SVM_Radial',
                      'SVM_Linear',
                      'SVM_sigmoid',
                      'Random_Forest',
                      'Redes_Neuronales',
                      'Boosting',
                      'XGboosting',
                      'Ensamble',
                      'Stacking'
)
resultado=round(resultado,2)
resultado

## Resultado Ordenado #####

# ordenamos por el Indicador que deseamos, quiza Accuracy en forma decreciente
Resultado_ordenado <- resultado[order(-Accuracy),] 
Resultado_ordenado

