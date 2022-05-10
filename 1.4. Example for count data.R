rm(list = ls())
library(dplyr)
library(SKM)

# Cambiar: Dirección de lo datos
Dir <- "C:/Users/Marina/Documents/Berna/Practicas profesionales/21Feb2022"
setwd(Dir)      #Fijar el directorio de los datos
name <- "Maize" #Ajustar al conjunto de datos Correspondiente
load(paste(name, "Toy.RData", sep = ""), verbose = TRUE) #Cargar los datos
ls();head(PhenoToy)

# Data preparation of Env, G & GE
Line <- model.matrix(~0 + Line, data = PhenoToy)
Env <- model.matrix(~0 + Env, data = PhenoToy)
Geno <- t(chol(GenoToy[, -1])) #First column is Line
LineG <- Line %*% Geno
LinexGenoxEnv <- model.matrix(~ 0 + LineG:Env)

# Predictor and Response Variables
X <- cbind(Env, LineG, LinexGenoxEnv)
y <- PhenoToy$PH
print(y[1:15])
typeof(y)

#Random Partition
set.seed(2022)
GIDs=unique(PhenoToy$Line)    #Unique Lines
folds <- cv_random(length(GIDs))

# A data frame that will contain the variables:
## (Number) Fold, Line, Env, (testing values) Observed, Predicted (values)
Predictions <- data.frame()
Hyperparams <- data.frame()

# Model training and predictions of the ith partition
for (i in seq(folds)) {
  cat("*** Fold:", i, " ***\n")
  # Identify the training and testing Line sets
  fold <- folds[[i]]
  Lines_sam_i=GIDs[fold$training]
  Lines_sam_i
  fold_i <-which(PhenoToy$Line %in% Lines_sam_i)
  length(fold_i)
  
  # Identify the training and testing sets
  X_training <- X[fold_i, ]
  X_testing <- X[-fold_i, ]
  y_training <- y[fold_i]
  y_testing <- y[-fold_i]
  
  #Optional: Removing columns with no variance
  var_x=apply(X_training,2,var)
  length(var_x)
  pos_var0=which(var_x>0)
  X_training=X_training[,pos_var0]
  X_testing=X_testing[,pos_var0]
  
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
  FoldPredictions <- data.frame(
    Fold = i,
    Line = PhenoToy$Line[-fold_i],
    Env = PhenoToy$Env[-fold_i],
    Observed = y_testing,
    Predicted = predictions$predicted)
  Predictions <- rbind(Predictions, FoldPredictions)
  
  # Hyperparams for the Fold
  HyperparamsFold <- model$hyperparams_grid %>%
    mutate(Fold = i)
  Hyperparams <- rbind(Hyperparams, HyperparamsFold)
  
  # Best hyperparams of the model
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
summaries$env[, 1:8]
summaries$env[, 9:15]
summaries$env[, 16:19]

#Summaries by Fold
summaries$fold[, 1:8]
summaries$fold[, 9:14]
summaries$fold[, 15:19]

# First rows of Hyperparams
head(Hyperparams)

# Last rows of Hyperparams
tail(Hyperparams)