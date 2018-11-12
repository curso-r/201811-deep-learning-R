input_descricao <- layer_input(shape = 30)
input_preco <- layer_input(shape = 1)
input_titulo <- layer_input(shape = 10)

embedding <- layer_embedding(input_dim = 1000, output_dim = 32)
lstm <- layer_lstm(units = 64)

vetor_descricao <- input_descricao %>% 
  embedding() %>% 
  lstm()

vetor_titulo <- input_titulo %>% 
  embedding() %>% 
  lstm()

output <- layer_concatenate(list(vetor_titulo, vetor_descricao, input_preco)) %>% 
  layer_dense(1, activation = "sigmoid")
