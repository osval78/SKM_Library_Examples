rm(list = ls())
library(dplyr)
library(SKM)

# Cambiar: Dirección de lo datos
Dir <- "C:/Users/Marina/Documents/Berna/Practicas profesionales/21Feb2022"
setwd(Dir)          #Fijar el directorio de los datos
name <- "Groundnut" #Ajustar al conjunto de datos Correspondiente
load(paste(name, "Toy.RData", sep = ""), verbose = TRUE) #Cargar los datos
ls()

# Data preparation of Env, G & GE
Line <- model.matrix(~0 + Line, data = PhenoToy)
Env <- model.matrix(~0 + Env, data = PhenoToy)
Geno <- cholesky(GenoToy[, -1]) #First column is Line
LinexGeno <- Line %*% Geno      #G matrix
LinexGenoxEnv <- model.matrix(~ 0 + LinexGeno:Env)

# Predictor and Response Variables
X <- cbind(Env, LinexGeno, LinexGenoxEnv)
y <- PhenoToy$PYPP

dim(X)
print(y[1:7])
typeof(y)

kernels <- c("Sparse_Gaussian", "Sparse_Arc_cosine")
lines_proportions <- c(0.5, 0.6, 0.7, 0.8, 0.9, 1)

# Example: Apply the "Sparse_Gaussian" Kenel to the data
kernels[1]
X_Linear <- kernelize(X, kernel = kernels[1], rows_proportion = lines_proportions[1])
# Note that X_Linear is an square matrix
dim(X_Linear)

# Empty lists that will contain Predictions, Times of execution & Summaries for each type of kernel
PredictionsAll <- list()
TimesAll <- list()
HyperparamsAll <- list()
SummariesAll <- list()

for (kernel in kernels) {
  cat("\n")
  cat("*** Kernel:", kernel, " ***\n")
  for (line_proportion in lines_proportions) {
    cat("\t*** Line_Proportion:", line_proportion, " ***\n")
    
    # Compute the kernel
    X <- kernelize(
      X,
      kernel = kernel,
      arc_cosine_deep = 2,
      rows_proportion = line_proportion
    )
    
    #Random Partition
    set.seed(2022)
    folds <- cv_random(
      records_number = nrow(X),
      folds_number = 5,
      testing_proportion = 0.2
    )
    
    # Empty data frames that will contain Predictions, Times of execution & Summaries for each partition
    Predictions <- data.frame()
    Times <- data.frame()
    Hyperparams <- data.frame()
    
    for (i in seq_along(folds)) {
      cat("\t*** Fold:", i, " ***\n")
      fold <- folds[[i]]
      
      # Identify the training and testing sets
      X_training <- X[fold$training, ]
      X_testing <- X[fold$testing, ]
      y_training <- y[fold$training]
      y_testing <- y[fold$testing]
      
      # Model training
      model <- generalized_linear_model(
        x = X_training,
        y = y_training,
        
        # Specify the hyperparameters values
        alpha = list(min = 0, max = 1),
        lambdas_number = 100,
        
        tune_folds_number = 5,
        
        tune_type = "bayesian_optimization",
        tune_bayes_samples_number = 5, 
        tune_bayes_iterations_number = 5,
        tune_grid_proportion = 0.8,
        
        #In this example the iterations wont be shown
        verbose = FALSE
      )
      
      # Testing Predictions
      predictions <- predict(model, X_testing)
      
      # Predictions for the Fold
      FoldPredictions <- data.frame(
        Fold = i,
        Line = PhenoToy$Line[fold$testing],
        Env = PhenoToy$Env[fold$testing],
        Observed = y_testing,
        Predicted = predictions$predicted
      )
      Predictions <- rbind(Predictions, FoldPredictions)
      
      # Execution times
      FoldTime <- data.frame(
        Kernel = kernel,
        Fold = i,
        Minutes = as.numeric(model$execution_time, units = "mins")
      )
      Times <- rbind(Times, FoldTime)
      
      # Hyperparams for the Fold
      HyperparamsFold <- model$hyperparams_grid %>%
        mutate(Fold = i)
      Hyperparams <- rbind(Hyperparams, HyperparamsFold)
    }
    # Sumaries of the Folds
    summaries <- gs_summaries(Predictions)
    
    # Predictions, Times of execution & Summaries for the specified Kernel
    PredictionsAll[[kernel]][[paste("Line_Proprtion:", line_proportion)]] <- Predictions
    TimesAll[[kernel]][[paste("Line_Proprtion:", line_proportion)]] <- Times
    HyperparamsAll[[kernel]][[paste("Line_Proprtion:", line_proportion)]]  <- Hyperparams
    SummariesAll[[kernel]][[paste("Line_Proprtion:", line_proportion)]] <- summaries
  }
}

# Predictios for theSparse_Gaussian Kernel
head(PredictionsAll$Sparse_Gaussian$`Line_Proprtion: 0.7`)

#Times of execution for the Sparse_Gaussian Kernel
TimesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`

# Elements of SummariesAll
names(SummariesAll)

# Elements of summaries for Sparse_Gaussian Kernel
names(SummariesAll$Sparse_Gaussian)
names(SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`)

# Summaries by Line
head(SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$line)

#Summaries by Enviroment
SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$env[, 1:8]
SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$env[, 9:15]
SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$env[, 16:19]

#Summaries by Fold
SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$fold[, 1:8]
SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$fold[, 9:15]
SummariesAll$Sparse_Gaussian$`Line_Proprtion: 0.7`$fold[, 16:19]

# First rows of Hyperparams
head(HyperparamsAll$Sparse_Gaussian$`Line_Proprtion: 0.7`)

# Last rows of Hyperparams
tail(HyperparamsAll$Sparse_Gaussian$`Line_Proprtion: 0.7`)