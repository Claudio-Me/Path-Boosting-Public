check.and.install.Package <- function(package_name) {
  if (!package_name %in% installed.packages()) {
    install.packages(package_name)
  }
  library(package_name)
}


# this function calls m boost and it returns the most important feature in the matrix and the fitted model
call_mboost <-
  function(matrix, labels, my_boost_control = boost_control()) {
    #convert matrix to data frame
    data <- as.data.frame(matrix)
    
    
    data$labels <- labels
    
    
    m_boost_model <-
      mboost(labels ~ .,
             data = data,
             baselearner = btree,
             control = my_boost_control)
    columns_importance <- varimp(m_boost_model)
    
    
    #needed do add-1 because python's arrays start from zero
    best_column = as.integer(which.max(columns_importance)) - 1
    
    
    
    return(list(column = best_column, model = m_boost_model))
    
  }



save_my_model