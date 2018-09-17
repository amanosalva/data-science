#Árboles de decisión

install.packages("smbinning", dependencies = T)
library(smbinning)
library(mlr)
library(ggplot2)

df = chileancredit
names(df)

#Exportando la data original
write.csv(df, "chileancredit.csv",row.names = FALSE)

data <- df 
names(data)

#En este caso, FlagGB es la variable Risk
#Además, esta data está dividida en train y test mediante la variable FlagSample (1:75%, 0:25%)

#Paso 1, dividimos la data en train y test mediante la variable FlagSample
data.train <- subset(data, FlagSample == 1)
data.test <- subset(data, FlagSample == 0)

#Exportamos la data tanto test como train
write.csv(data.train, "train.csv", row.names = F)
write.csv(data.test, "test.csv", row.names = F)

#trabajaremos sobre la data train
str(data.train)

summarizeColumns(data.train)
#Las siguientes variables tienen valores perdidos:
# TOB - 587
# IncomeLevel - 273
# MaxDqBin02 - 186
# MaxDqBin03 - 256
# MaxDqBin04 - 340
# MaxDqBin05 - 403
# MaxDqBin06 - 476
# FlagGB - 1068

#La variable target tiene valores perdidos!

#Veamos el análisis exploratorio, creando gráficos.

#Gráfico de dispersión: Relación entre variables, y como cierta cantidad de variables puede influir en otras.
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_point()

#Con estilos:
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_point(alpha = 0.5, size = 5, aes(color = factor(FlagGB)))

#Con una recta lineal aproximada
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_point(alpha = 0.5, size = 5, aes(color = factor(FlagGB))) + 
  geom_smooth(method = "lm", se = FALSE, col = "green") +
  facet_grid(FlagGB~.)

#Sacando la recta a cada grupo según el riesgo
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_point(alpha = 0.5, size = 5, aes(color = factor(FlagGB))) + 
  geom_smooth(method = "lm", se = FALSE, col = "green") +
  facet_grid(FlagGB~.)
 
# Cambiando de fuente de letra 
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_point(alpha = 0.5, size = 5, aes(color = factor(FlagGB))) + 
  geom_smooth(method = "lm", se = FALSE, col = "green") +
  facet_grid(FlagGB~.) +
  theme_bw(base_family = "Times", base_size = 10)

#Añadiendo etiquetas en el eje X
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_point(alpha = 0.5, size = 5, aes(color = factor(FlagGB))) + 
  geom_smooth(method = "lm", se = FALSE, col = "green") +
  facet_grid(FlagGB~.) +
  theme_bw(base_family = "Times", base_size = 10) + 
  labs(x = "Saldo pendiente de la hipoteca") + labs(y = "Saldo Pendiente") + labs(title = "Saldo Pendiente vs Saldo pendiente de la hipóteca")


#### Gráficos de líneas
plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_line(linetype = "dashed", color = "red")


plot <- ggplot(data.train, aes(MtgBal01, Bal01))
plot + geom_line(aes(color = as.factor(FlagGB))) + 
labs(x = "Saldo pendiente de la hipoteca") + labs(y = "Saldo Pendiente") + labs(title = "Saldo Pendiente vs Saldo pendiente de la hipóteca")



#### Gráfico de barras
# Para esto tenemos que usar el paquete dplyr para hacer un pequeño dataframe que las barras puedan comprender
install.packages("dplyr", dependencies = T)
library(dplyr)
data.train.sum <- data.train %>% group_by(FlagGB = as.factor(FlagGB), IncomeLevel = IncomeLevel) %>% count(FlagGB)
data.train.sum

plot <- ggplot(data.train.sum, aes(FlagGB, n, fill = IncomeLevel, label = scales::comma(n)))
plot + geom_bar(show.legend = T, stat = "identity") +
  labs(title = "Riesgo")
