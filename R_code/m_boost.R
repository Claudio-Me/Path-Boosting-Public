initialize_environment <- function(R_code_location) {
  # set working directory as the same of this file
  # Install the this.path package.
  # if (!'this.path' %in% installed.packages()) {
  #   install.packages('this.path')
  # }
  
  # library(this.path)
  
  # Get the directory from the path of the current file.
  # cur_dir2 = this.path()
  cur_dir2 = "C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/R_code/m_boost.R"
  
  
  
  
  
  # Set the working directory.
  # setwd(cur_dir2)
  R_code_location = paste(R_code_location, "/", "my_R_functions.R", sep = "")
  source(R_code_location)
  
  
  #load mboost
  #install.packages("C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/R_code/my_mboost", repos = NULL, type = "source")
  check.and.install.load.Package("mboost")
  
  
}



first_iteration <-
  function(data_matrix,
           y,
           model_name,
           model_location,
           family_name,
           base_learner_name) {
    # it does just one iteration over the original data and initialize all the files needed
    
    y <- as.numeric(y)
    initialize_environment(model_location)
    
    # print("matrix:   ")
    # print(matrix)
    # print(typeof(matrix))
    # print("Y:   ")
    # print(y)
    # print(typeof(y))
    
    
    column_and_model = fit_mboost(
      data_matrix,
      y,
      family_name,
      my_boost_control = boost_control(mstop = 1),
      base_learner = base_learner_name
    )
    
    
    selected_column = column_and_model$column
    
    trained_model = column_and_model$model
    
    
    # location of the model is in the same directory as the location of this script
    model_location = paste(model_location, "/", model_name, ".RData", sep = "")
    
    # we save the model on a file, it has to be a list because of the function "select_column"
    base_learners_list = list(trained_model)
    save(base_learners_list, file = model_location)
    
    
    # print("selected column:")
    # print(selected_column)
    
    
    
    return (selected_column)
    
    
  }


select_column <-
  function(data_matrix,
           y,
           model_name,
           model_location,
           family_name,
           base_learner_name) {
    # it does just one iteration over the original data and initialize all the files needed
    # -------------------------------------------------------------------------------------
    
    
    print("start_function")
    
    #--------------------------------------------------------------------------------------
    y <- as.numeric(y)
    initialize_environment(model_location)
    
    #print("matrix:   ")
    #print(matrix)
    #print(typeof(matrix))
    #print("Y:   ")
    #print(y)
    #print(typeof(y))
    
    # location of the model is in the same directory as the location of this script
    model_location = paste(model_location, "/", model_name, ".RData", sep = "")
    
    # -------------------------------------------------------------------------------------
    
    
    print("load model")
    
    #--------------------------------------------------------------------------------------
    
    # load previous models
    load(model_location)
    
    data_frame_matrix = as.data.frame(data_matrix)
    
    
    # -------------------------------------------------------------------------------------
    
    
    print("get past prediction")
    
    #--------------------------------------------------------------------------------------
    
    # compute the new target "gradient" using the negative gradient
    
    #remember if dataframe is null, then it evaluates the model using the data used for training
    predictions_vector = sapply(base_learners_list, predict_mboost, data_frame_matrix = NULL)
    y_hat = rowSums(predictions_vector)
    
    # gradient of the loss function:
    first_base_learner = base_learners_list[[1]]
    negative_gradient_function = slot(first_base_learner$family, "ngradient")
    
    
    # -------------------------------------------------------------------------------------
    
    
    print("compute gradient")
    
    #--------------------------------------------------------------------------------------
    
    gradient = negative_gradient_function(y, y_hat)
    
    
    
    # -------------------------------------------------------------------------------------
    
    
    print("fit new learner")
    
    #--------------------------------------------------------------------------------------
    
    
    column_and_model = fit_mboost(
      data_frame_matrix,
      gradient,
      family_name,
      my_boost_control = boost_control(mstop = 1),
      base_learner = base_learner_name
    )
    
    selected_column = column_and_model$column
    
    trained_model = column_and_model$model
    
    base_learners_list = append(base_learners_list, list(trained_model))
    
    
    save(base_learners_list, file = model_location)
    
    
    # -------------------------------------------------------------------------------------
    
    
    print("return ")
    
    #--------------------------------------------------------------------------------------
    
    rm(list=setdiff(ls(), "selected_column"))
    
    gc()
    
    
    
    return (selected_column)
    
    
    
  }


main_predict <- function(data_matrix, model_name, model_location) {
  initialize_environment(model_location)
  # location of the model is in the same directory as the location of this script
  model_location = paste(model_location, "/", model_name, ".RData", sep = "")
  
  # load the model
  load(model_location)
  
  data_frame_matrix = as.data.frame(data_matrix)
  
  predictions_vector = sapply(base_learners_list, predict_mboost, data_frame_matrix = data_frame_matrix)
  predictions_vector = rowSums(predictions_vector)
  
  
  
  return(predictions_vector)
  
}



test <- function() {
  model_location = "C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/R_code"
  initialize_environment(model_location)
  n_observations = 34
  n_colums = 30
  base_learner_name = "bbs"
  family_name = "Gaussian"
  
  mat1.data <- runif(n_observations * n_colums)
  mat1 <- matrix(mat1.data,
                 nrow = n_observations,
                 ncol = n_colums,
                 byrow = TRUE)
  
  y = runif(n_observations)
  
  
  result1 = first_iteration(
    mat1,
    y,
    "test",
    model_location,
    family_name = family_name,
    base_learner_name = base_learner_name
  )
  print("First Column:")
  print(result1)
  
  
  
  mat2 <- matrix(mat1.data,
                 nrow = n_observations,
                 ncol = n_colums + 2,
                 byrow = TRUE)
  result2 = select_column(
    mat2,
    y,
    "test",
    model_location,
    family_name = family_name,
    base_learner_name = base_learner_name
  )
  print("second_column:")
  print(result2)
  
  return(c(result1, result2))
}

