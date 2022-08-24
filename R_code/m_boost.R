initialize_environment <- function() {
  # set working directory as the same of this file
  # Install the this.path package.
  #if(!'this.path'%in%installed.packages('this.path')){
  #  install.packages('this.path')
  #}
  
  
  
  # Get the directory from the path of the current file.
  cur_dir2 = this.dir()
  
  # Set the working directory.
  setwd(cur_dir2)
  
  # Check that the working directory has been set as desired.
  getwd()
  source("my_R_functions.R")
  
  #load mboost
  #install.packages("C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/R_code/my_mboost", repos = NULL, type = "source")
  check.install.load.Package("mboost")
}

test <- function() {
  mat1.data <- c(1, 0, 0, 1)
  mat1 <- matrix(mat1.data,
                 nrow = 2,
                 ncol = 2,
                 byrow = TRUE)
  y = c(0.08806, 0.05682)
  
  
  result = select_column(mat1, y)
  print("Result:")
  print(result)
  return(result)
}

first_iteration <- function(matrix, y, model_name) {
  # it does just one iteration over the original data and initialize all the files needed
  
  y <- as.numeric(y)
  initialize_environment()
  print("matrix:   ")
  print(matrix)
  print(typeof(matrix))
  print("Y:   ")
  print(y)
  print(typeof(y))
  
  
  
  column_and_model = call_mboost(matrix, y, my_boost_control = boost_control(mstop = 1))
  
  selected_column = column_and_model$column
  
  trained_model = column_and_model$model
  
  result[1]
  model_location = paste(this.dir(), "model.RData", sep = "")
  
  save(trained_model, file = model_location)
  
  
  return(list(selected_column, model_location))
  
  
}

select_column <- function(matrix, y, model_location) {
  y <- as.numeric(y)
  initialize_environment()
  print("matrix:   ")
  print(matrix)
  print(typeof(matrix))
  print("Y:   ")
  print(y)
  print(typeof(y))
  
  column_and_model$column = call_mboost(matrix, y)
  selected_column = column_and_model$column
  
  trained_model = column_and_model$model
  
  #save the model in a file
  model_location = paste(this.dir(), "model.RData", sep = "")
  save(trained_model, file = model_location)
  
  
  return(list(selected_column, model_location))
  

  
  
}

result = test()
result = result[[2]]

tmp1 = result[1]
tmp2 = result$baselearner
tmp3 = tmp1$fitted()
result[1]$fitted()


tmp = result[1]$fitted()

save(tmp, file = paste(this.dir(), "model.RData", sep = ""))

load("test.RData")
