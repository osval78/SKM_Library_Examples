rm(list = ls())
library(dplyr)
library(SKM)

# Cambiar: Dirección de lo datos
Dir <- "C:/Users/Marina/Documents/Berna/Practicas profesionales/21Feb2022"
setwd(Dir)      #Fijar el directorio de los datos
name <- "Maize" #Ajustar al conjunto de datos Correspondiente
load(paste(name, "Toy.RData", sep = ""), verbose = TRUE) #Cargar los datos
ls()

# Data preparation of Env & G
Line <- model.matrix(~0 + Line, data = PhenoToy)
Env <- model.matrix(~0 + Env, data = PhenoToy)
Geno <- t(chol(GenoToy[, -1])) #First column is Line
LineG <- Line %*% Geno         #G matrix

# Predictor and Response Variables
X <- cbind(Env, LineG)
y <- PhenoToy[, c("Yield", "ASI")]

# 7-Fold partition
set.seed(2022)
folds <- cv_kfold(
  records_number = nrow(X),
  k = 7
)

# Data frames that will contain the variables:
## (Number) Fold, Line, Env, (testing values) Observed, Predicted (values)
PredictionsYield <- data.frame()
PredictionsASI <- data.frame()
Hyperparams <- data.frame()

# Model training and predictions of the ith partition
for (i in seq_along(folds)) {
  cat("*** Fold:", i, " ***\n")
  #i=1
  fold <- folds[[i]]
  
  # Identify the training and testing sets
  X_training <- X[fold$training, ]
  X_testing <- X[fold$testing, ]
  y_training <- y[fold$training,]
  y_testing <- y[fold$testing,]
  
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
  
  # Predictions of Yield for the Fold
  FoldPredictionsYield <- data.frame(
    Fold = i,
    Line = PhenoToy$Line[fold$testing],
    Env = PhenoToy$Env[fold$testing],
    Observed = y_testing$Yield,
    Predicted = predictions$Yield$predicted
  )
  PredictionsYield <- rbind(PredictionsYield, FoldPredictionsYield)
  
  # Predictions of ASI for the Fold
  FoldPredictionsASI <- data.frame(
    Fold = i,
    Line = PhenoToy$Line[fold$testing],
    Env = PhenoToy$Env[fold$testing],
    Observed = y_testing$ASI,
    Predicted = predictions$ASI$predicted
  )
  PredictionsASI <- rbind(PredictionsASI, FoldPredictionsASI)
  
  # Hyperparams for the Fold
  HyperparamsFold <- model$hyperparams_grid %>%
    mutate(Fold = i)
  Hyperparams <- rbind(Hyperparams, HyperparamsFold)
  
  # Best hyperparams of the model
  print(model$best_hyperparams)
}

head(PredictionsASI)
unique(PredictionsASI$Fold)
head(PredictionsYield)
unique(PredictionsYield$Fold)

# Summaries
summariesASI <- gs_summaries(PredictionsASI) 
summariesYield <- gs_summaries(PredictionsYield) 

# Elements of summaries
names(summariesASI)

# Summaries by Line
head(summariesASI$line)
head(summariesYield$line)

#Summaries by Enviroment
summariesASI$env[,1:9]
summariesYield$env[,1:9]

#Summaries by Fold
summariesASI$fold[,1:8]
summariesYield$fold[,1:8]

# First rows of Hyperparams
head(Hyperparams)

# Last rows of Hyperparams
tail(Hyperparams)