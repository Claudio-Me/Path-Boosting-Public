initialize_environment <- function() {
  # set working directory as the same of this file
  # Install the this.path package.
  if (!'this.path' %in% installed.packages()) {
    install.packages('this.path')
  }
  library(this.path)
  # Get the directory from the path of the current file.
  cur_dir2 = this.dir()
  
  # Set the working directory.
  setwd(cur_dir2)
  
  # Check that the working directory has been set as desired.
  getwd()
  source("my_R_functions.R")
  
  #load mboost
  #install.packages("C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/R_code/my_mboost", repos = NULL, type = "source")
  check.and.install.load.Package("mboost")
}



first_iteration <- function(matrix, y, model_name, family_name) {
  # it does just one iteration over the original data and initialize all the files needed
  
  y <- as.numeric(y)
  initialize_environment()
  print("matrix:   ")
  print(matrix)
  print(typeof(matrix))
  print("Y:   ")
  print(y)
  print(typeof(y))
  
  
  
  column_and_model = call_mboost(matrix, y, my_boost_control = boost_control(mstop = 1), family_name)
  
  selected_column = column_and_model$column
  
  trained_model = column_and_model$model
  
  
  # location of the model is in the same directory as the location of this script
  model_location = paste(this.dir(), "/", model_name, ".RData", sep = "")
  
  #we save the model on a file, it has to be a list because of the function "select_column"
  base_learners_list = list(trained_model)
  save(base_learners_list, file = model_location)
  
  
  return (selected_column)
  
  
}

select_column <- function(matrix, y, model_name, family_name) {
  # it does just one iteration over the original data and initialize all the files needed
  
  y <- as.numeric(y)
  initialize_environment()
  print("matrix:   ")
  print(matrix)
  print(typeof(matrix))
  print("Y:   ")
  print(y)
  print(typeof(y))
  
  # location of the model is in the same directory as the location of this script
  model_location = paste(this.dir(), "/", model_name, ".RData", sep = "")
  
  # load previous models
  load(model_location)
  
  # compute the new target using the negative gradient
  # gradient of the loss function:
  first_base_learner = base_learners_list[[1]]
  negative_gradient = slot(first_base_learner$family, "ngradient")
  #to do: compute negative gradient only on the first n column of matrix x
  
  
  column_and_model = call_mboost(matrix, y, my_boost_control = boost_control(mstop = 1), family_name)
  
  selected_column = column_and_model$column
  
  trained_model = column_and_model$model
  
  
  # location of the model is in the same directory as the location of this script
  model_location = paste(this.dir(), "model.RData", sep = "")
  
  
  save(base_learner, file = model_location)
  
  
  return (selected_column)
  
  
}


test <- function() {
  mat1.data <- c(1, 0, 0, 0, 1, 0, 0, 0, 1)
  mat1 <- matrix(mat1.data,
                 nrow = 3,
                 ncol = 3,
                 byrow = TRUE)
  y = c(0.08806, 0.05682, 0.34234)
  
  
  result = select_column(mat1, y, "test", "Gaussian")
  print("Result:")
  print(result)
  return(result)
}

result = test()

trained_model = result[[2]]
base_learner = trained_model[[1]]




tmp3 = tmp1$fitted()
result[1]$fitted()


tmp = result[1]$fitted()

save(tmp, file = paste(this.dir(), "model.RData", sep = ""))

load("test.RData")
