rm(list = ls())
library(dplyr)
library(SKM)

# Cambiar: Dirección de lo datos
Dir <- "C:/Users/Marina/Documents/Berna/Practicas profesionales/21Feb2022"
setwd(Dir)
name <- "EYT" #Ajustar al conjunto de datos Correspondiente
load(paste(name, "Toy.RData", sep = ""), verbose = TRUE)
ls()

# Data preparation of Env, G & GE
Line <- model.matrix(~0 + Line, data = PhenoToy)
Env <- model.matrix(~0 + Env, data = PhenoToy)
Geno <- t(chol(GenoToy[, -1])) #First column is Line
LineG <- Line %*% Geno         #G matrix
LinexGenoxEnv <- model.matrix(~ 0 + LineG:Env)    

# Predictor and Response Variables
X <- cbind(Env, LineG, LinexGenoxEnv)
y <- as.factor(ntile(PhenoToy$GY,3))
# First 30 responses
print(y[1:30])
# Changing the levels 
levels(y) <- c("A", "B", "C")
print(y[1:30])

#Random Partition
set.seed(2022)
folds <- cv_random(records_number = nrow(X))

# A data frame that will contain the variables:
## (Number) Fold, Line, Env, (testing values) Observed, Predicted (values) and the predicted probabilities of responses
Predictions <- data.frame()
Hyperparams <- data.frame()

# Model training and predictions of the ith partition
for (i in seq_along(folds)) {
  cat("*** Fold:", i, " ***\n")
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
    
    # Specify the hyperparameters ranges
    alpha = list(min = 0, max = 1),
    lambdas_number = 100,
    
    tune_type = "Bayesian_optimization",
    tune_bayes_samples_number = 5, 
    tune_bayes_iterations_number = 5
  )
  
  # Testing Predictions
  predictions <- predict(model, X_testing)
  
  # Predictions for the Fold
  FoldPredictions <- cbind(
    data.frame(
      Fold = i,
      Line = PhenoToy$Line[fold$testing],
      Env = PhenoToy$Env[fold$testing],
      Observed = y_testing,
      Predicted = predictions$predicted
    ), 
    predictions$probabilities)
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
summaries$env

#Summaries by Fold
summaries$fold

# First rows of Hyperparams
head(Hyperparams)

# Last rows of Hyperparams
tail(Hyperparams)