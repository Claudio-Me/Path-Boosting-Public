check.and.install.load.Package <- function(package_name) {
  if (!package_name %in% installed.packages()) {
    install.packages(package_name)
  }
  library(package_name, character.only = TRUE)
}

select_family <- function(family_name) {
  if (family_name == "Gaussian") {
    return (Gaussian())
  }  else if (family_name == "AdaExp") {
    return (AdaExp())
    
  }  else if (family_name == "AUC") {
    return (AUC())
    
  }  else if (family_name == "Binomial") {
    return (Binomial(
      type = c("adaboost", "glm"),
      link = c("logit", "probit", "cloglog", "cauchit", "log"),
      ...
    ))
    
  }  else if (family_name == "GaussClass") {
    return (GaussClass())
    
  }  else if (family_name == "GaussReg") {
    return (GaussReg())
    
  }  else if (family_name == "Huber") {
    return (Huber(d = NULL))
    
  }  else if (family_name == "Laplace") {
    return (Laplace())
    
  }  else if (family_name == "Poisson") {
    return (Poisson())
    
  }  else if (family_name == "GammaReg") {
    return (GammaReg(nuirange = c(0, 100)))
    
  }
  else if (family_name == "CoxPH") {
    return (CoxPH())
    
  }  else if (family_name == "QuantReg") {
    return (QuantReg(tau = 0.5, qoffset = 0.5))
    
  }  else if (family_name == "ExpectReg") {
    return (ExpectReg(tau = 0.5))
    
  }  else if (family_name == "NBinomial") {
    return (NBinomial(nuirange = c(0, 100)))
    
  }  else if (family_name == "PropOdds") {
    return (PropOdds(nuirange = c(-0.5,-1), offrange = c(-5, 5)))
    
  }  else if (family_name == "Weibull") {
    return (Weibull(nuirange = c(0, 100)))
    
  }  else if (family_name == "Loglog") {
    return (Loglog(nuirange = c(0, 100)))
    
  }  else if (family_name == "Lognormal") {
    return ((nuirange = c(0, 100)))
    
  }  else if (family_name == "Gehan") {
    return (Gehan())
    
  }  else if (family_name == "Hurdle") {
    return (Hurdle(nuirange = c(0, 100)))
    
  }  else if (family_name == "Multinomial") {
    return (Multinomial())
    
  }  else if (family_name == "Cindex") {
    return (Cindex(sigma = 0.1, ipcw = 1))
    
  }  else if (family_name == "CoxPH") {
    return (Gaussian())
    
  }  else if (family_name == "RCG") {
    return (RCG(nuirange = c(0, 1), offrange = c(-5, 5)))
  } else {
    stop('Family not recognized')
  }
  
  
  
  
}

get_number_of_columns_the_model_takes_in_input <- function(model) {
  return (length(model$basemodel))
}

predict_mboost <- function(model, matrix) {
  #convert matrix to data frame
  data <- as.data.frame(matrix)
  n_columns = get_number_of_columns_the_model_takes_in_input(model)
  
  matrix = matrix[, 1:n_columns]
  
  return (model$predict(matrix))
}


fit_mboost <-
  function(matrix,
           labels,
           my_boost_control = boost_control(),
           family_name) {
    # this function calls m boost and it returns the most important feature in the m,atrix and the fitted model
    
    
    #convert family name (string) to family object
    family_o = select_family(family_name)
    
    #convert matrix to data frame
    data <- as.data.frame(matrix)
    
    
    data$labels <- labels
    
    
    m_boost_model <-
      mboost(
        labels ~ .,
        data = data,
        baselearner = btree,
        control = my_boost_control,
        family = family_o
      )
    columns_importance <- varimp(m_boost_model)
    
    
    #needed do add-1 because python's arrays start from zero
    best_column = as.integer(which.max(columns_importance)) - 1
    
    
    
    return(list(column = best_column, model = m_boost_model))
    
  }
