rm(list = ls())
library(dplyr)
library(SKM)

# Cambiar: Dirección de lo datos
Dir <- "C:/Users/Marina/Documents/Berna/Practicas profesionales/21Feb2022"
setwd(Dir)          #Fijar el directorio de los datos
name <- "Groundnut" #Ajustar al conjunto de datos Correspondiente
load(paste(name, "Toy.RData", sep = ""), verbose = TRUE) #Cargar los datos

# Data preparation of G
Line <- model.matrix(~0 + Line, data = PhenoToy) 
Geno <- t(chol(GenoToy[, -1])) #First column is Line
LineG <- Line %*% Geno         #G matrix

# Predictor and Response Variables
X <- LineG
y <- PhenoToy$YPH

# Note that y is a continuous numeric vector
class(y)
typeof(y)

#Random Partition
set.seed(2022)
folds <- cv_random(records_number = nrow(X))

# A data frame that will contain the variables:
## (Number) Fold, Line, Env, (testing values) Observed and Predicted (values)
Predictions <- data.frame()
Hyperparams <- data.frame()

# Model training and predictions of the ith partition
for (i in seq_along(folds)) {
  cat("*** Fold:", i, " ***\n")
  fold <- folds[[i]]
  
  #Identify the training and testing sets
  X_training <- X[fold$training, ]
  X_testing <- X[fold$testing, ]
  y_training <- y[fold$training]
  y_testing <- y[fold$testing]
  
  # Model training
  model <- generalized_linear_model(
    x = X_training,
    y = y_training,
    
    # Specify the hyperparameters
    alpha = c(0, 0.25, 0.50, 0.75, 1),
    lambdas_number = 100,
    
    tune_type = "Grid_search"
  )
  
  #Prediction of the testing set
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
  
  # Hyperparams for the Fold
  HyperparamsFold <- model$hyperparams_grid %>%
    mutate(Fold = i)
  Hyperparams <- rbind(Hyperparams, HyperparamsFold)
  
  # Best hyperparams of the model
  cat("*** Optimal hyperparameters: *** \n")
  print(model$best_hyperparams)
}

head(Predictions)
unique(Predictions$Fold)

# Summaries
summaries <- gs_summaries(Predictions) 

# Elements of summaries
names(summaries)

# Summaries by Line
head(summaries$line)

#Summaries by Enviroment
summaries$env[,1:7]
summaries$env[,8:14]
summaries$env[,15:19]

#Summaries by Fold
summaries$fold[,1:8]

# First rows of Hyperparams
head(Hyperparams)

# Last rows of Hyperparams
tail(Hyperparams)