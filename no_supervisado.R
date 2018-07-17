rm(list=ls())

#########################################################################
### -- Machine Learning for Data Science -- ## 
#########################################################################

# ######### 1) LIBRERIAS A UTILIZAR ################# 
library(advclust)

# Cargar la data a utilizar
data_wine_red<-read.csv('winequality-red.csv', sep=";",header = TRUE)

# ver el optimo numero de cluster a formar

#calcula la suma total de cuadrados
wss <- (nrow(data_wine_red)-1)*sum(apply(scale(data_wine_red[,1:11]),2,var))
#la calcula por clusters
for (i in 2:9) wss[i] <- sum(kmeans(scale(data_wine_red[,1:11]),
                                    centers=i)$withinss)
plot(1:9, wss, type="b", xlab="Nummero de Clusters",
     ylab="Suma de cuadrados within") 

# kmeans

kmean<-kmeans(scale(data_wine_red[,1:11]),centers=6,iter.max=20)
kmean$size
kmean$withinss

# Fuzzy C-Means

fuzzy.CM(X=data_wine_red[,1:11],K = 6,m = 2,RandomNumber = 1234)->cl_CM
print(cl_CM)
table(cl_CM@hard.label)

biploting(cl_CM, data_wine_red[,1:11], scale=T)->biplot

# Gustafson Kessel

fuzzy.GK(X=data_wine_red[,1:11],K = 6,m = 2,RandomNumber = 1234)->cl_GK
show(cl_GK)
table(cl_GK@hard.label)

# Gath Geva

fuzzy.GG(X=data_wine_red[,1:11],K = 6,m = 2,RandomNumber = 1234)->cl_GG
show(cl_GG)
table(cl_GG@hard.label)

# Consensus Clustering

c_fuzzycluster(cl_GK,cl_GG,cl_CM)->c_consensus
co.vote(c_consensus,"sum")

# KOHONEN

# LIBRERIAS A UTILIZAR
library(kohonen)

# ENTENDIMIENTO DE LA LOGICA

# Se calcula una distancia ponderada sobre todas las capas 
# para determinar las unidades ganadoras durante el entrenamiento. 
# Las funciones som y xyf son simplemente envolturas para superestructuras 
# con una y dos capas, respectivamente

#USO DE LA FUNCIÓN SOM DE VINOS
datos=read.table("VINOS.txt",header=T)

set.seed(7)
vinos.sc=scale(datos[,1:13])
vino.som=som(vinos.sc, grid = somgrid(5,5,"hexagonal"))
names(vino.som)
summary(vino.som)
vino.som$unit.classif #unidades ganadoras para todos los objetos de datos
vino.som$codes #una lista de matrices que contienen vectores de libro de códigos.
plot(vino.som, main="Datos de vino")

#USO DE LA FUNCIÓN xyf DE VINOS
attach(datos)
set.seed(7)
kohmap = xyf(vinos.sc, classvec2classmat(clase),grid = somgrid(5, 5, "hexagonal"), rlen=100)
plot(kohmap,type="codes",main=c("Distribución de variables","Clases de códigos"))
plot(kohmap,type="mapping",,col=clase+1,main="Mapa de clases")
plot(kohmap,type="mapping",labels=clase,col=clase+1,main="Mapa de clases")
plot(kohmap,type="counts",main="Diagrama de conteos")
plot(kohmap,type="quality",labels=clase,col=clase+1,main="Mapa de calidad")