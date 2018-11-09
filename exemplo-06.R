# Pacotes -----------------------------------------------------------------

library(keras)
library(tidyverse)

# Dados -------------------------------------------------------------------

imdb <- dataset_imdb()
imdb_words <- dataset_imdb_word_index()
words <- imdb_words %>% imap_dfr(~dplyr::data_frame(id = .x, palavra = .y))

imdb$train$x[[1]] %>% 
  data_frame(id = . - 3) %>% 
  left_join(words, by = "id") %>% 
  with(palavra) %>% 
  paste(collapse = " ")

x_train <- imdb$train$x %>%
  pad_sequences(maxlen = 400)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = 400)