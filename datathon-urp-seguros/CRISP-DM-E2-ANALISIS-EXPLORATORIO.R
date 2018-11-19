install.packages('DataExplorer', dependencies = T) 
library(mlr)
library(dplyr)

#Análisis exploratorio:

train <- read.csv("train_seguros.csv", header = T)

#Graficando los valores faltantes para una mejor visión global:
PlotMissing(train)

# En caso de querer saber los valores exactos de los NA:
summarizeColumns(train)
# Las siguientes columnas tienen valores NA:
# Antiguedad_Maxima - 587
# edad_Maxima - 273
# Puntaje_MorosidadX, donde 1 <= x <= 5

# Viendo la estructura de la data:
PlotStr(train)
# Curiosamente vemos que la variable "Siniestro6" tiene tres niveles: "", "si", "no". Ese espacio en blanco
# debe ser considerado como ata faltante? Habría que ver su proporción o influencia.
levels(train$Siniestros6)


# Viendo las distribuciones numéricas:
## Separamos solo los numéricos
train_numericos <- select_if(train, is.numeric)


names(train_numericos)

HistogramContinuous(train_numericos[1:9])
HistogramContinuous(train_numericos[10:length(train_numericos)])

# Viendo las distribuciones categóricas:
train_factores <- select_if(train, is.factor)
names(train_factores)

BarDiscrete(train_factores)


# Viendo las correlaciones continuas:
CorrelationContinuous(train_numericos)

# Viendo las correlaciones discretas:
CorrelationDiscrete(train_factores)



GenerateReport(train)



?DataExplorer
