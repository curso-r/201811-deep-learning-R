# Pacotes -----------------------------------------------------------------

library(keras)
library(Metrics)
library(ggplot2)

# Gerando dados -----------------------------------------------------------

n_row <- 1000
n_col <- 10

X <- rnorm(n_row*n_col) %>% matrix(nrow = n_row)
y <- apply(X, 1, sum) + rnorm(n_row)


# Algumas visualizações ---------------------------------------------------

qplot(X[,3], y) + geom_smooth(method = "lm")
modelo <- lm(y ~ 0 + X)

# Definindo o modelo ------------------------------------------------------

input <- layer_input(shape = n_col)
output <- layer_dense(input, units = 1, use_bias = FALSE)

modelo <- keras_model(input, output)
summary(modelo)


# Gerando previsões -------------------------------------------------------

y_hat <- predict(modelo, X)
dim(y_hat)
qplot(y, y_hat[,1], geom  = "point")


# Calculando o erro -------------------------------------------------------

rmse(y, y_hat)

# Compilando o modelo -----------------------------------------------------

# Mas e agora, precisamos treinar o modelo
# Mas antes precisamos definir a função de perda, e o otimizador


custom_mae <- custom_metric("custom_mae", function(y_true, y_pred) {
  k_mean(k_abs(y_true - y_pred))
})

custom_mse <- function(y_true, y_pred) {
  k_mean(k_pow(y_true - y_pred, 2))
}

modelo %>% 
  compile(
    loss = custom_mse,
    optimizer = optimizer_sgd(lr = 0.01), 
    metrics = custom_mae
  )


# Ajustando os modelos ----------------------------------------------------

# Agora podemos ajustar
modelo %>% 
  fit(X , y, validation_split = 0.2, 
      batch_size = 32, epochs = 10)


# Calculando o erro novamente ---------------------------------------------

# Podemos prever de novo:

y_hat <- predict(modelo, X)
rmse(y, y_hat)
