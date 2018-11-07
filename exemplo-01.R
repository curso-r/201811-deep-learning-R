# Pacotes -----------------------------------------------------------------

library(keras)
library(Metrics)

# Gerando dados -----------------------------------------------------------

n_row <- 1000
n_col <- 10

X <- rnorm(n_row*n_col) %>% matrix(nrow = n_row)
y <- apply(X, 1, sum) + rnorm(n_row)


# Algumas visualizações ---------------------------------------------------

qplot(X[,1], y) + geom_smooth(method = "lm")


# Definindo o modelo ------------------------------------------------------

input <- layer_input(n_col)
output <- layer_dense(input, units = 1)

modelo <- keras_model(input, output)
summary(modelo)


# Gerando previsões -------------------------------------------------------

y_hat <- predict(modelo, X)
dim(y_hat)


# Calculando o erro -------------------------------------------------------

rmse(y, y_hat)

# Compilando o modelo -----------------------------------------------------

# Mas e agora, precisamos treinar o modelo
# Mas antes precisamos definir a função de perda, e o otimizador

modelo %>% 
  compile(
    loss = "mse",
    optimizer = "sgd"
  )


# Ajustando os modelos ----------------------------------------------------

# Agora podemos ajustar
modelo %>% 
  fit(X , y, validation_split = 0.2)


# Calculando o erro novamente ---------------------------------------------

# Podemos prever de novo:

y_hat <- predict(modelo, X)
rmse(y, y_hat)
