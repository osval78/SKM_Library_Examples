rm(list = ls())
library(dplyr)
library(SKM)

# Cambiar: Dirección de lo datos
Dir <- "C:/Users/Marina/Documents/Berna/Practicas profesionales/21Feb2022"
setwd(Dir)          #Fijar el directorio de los datos
name <- "Chickpea" #Ajustar al conjunto de datos Correspondiente
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
y <- PhenoToy$Biomass

dim(X)
print(y[1:7])
typeof(y)

kernels <- c(
  "Linear",
  "Polynomial",
  "Sigmoid",
  "Gaussian",
  "Exponential",
  "Arc_cosine",
  "Arc_cosine_L"
)

# Example: Apply the Linear Kenel to the data
kernels[1]
X_Linear <- kernelize(X, kernel = kernels[1])
# Note that X_Linear is an square matrix
dim(X_Linear)

# Empty lists that will contain Predictions, Times of execution & Summaries for each type of kernel
PredictionsAll <- list()
TimesAll <- list()
HyperparamsAll <- list()
SummariesAll <- list()

# Model training for each type of kernel
for (kernel in kernels) {
  cat("\n")
  cat("*** Kernel:", kernel, " ***\n")
  
  # Identify the arc_deep and the kernel 
  arc_deep <- 2
  if (kernel == "Arc_cosine_L") {
    arc_deep <- 3
    ckernel <- "Arc_cosine"
  } else {
    ckernel <- kernel
  }
  
  # Compute the kernel
  X <- kernelize(X, kernel = ckernel, arc_cosine_deep = arc_deep)
  
  # Random Partition
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
      alpha = c(0, 0.25, 0.50, 0.75, 1),
      lambdas_number = 100,
      
      tune_folds_number = 5,
      
      tune_type = "grid_search",
      tune_grid_proportion = 0.8,
      
      # In this example the iterations wont be shown
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
  PredictionsAll[[kernel]] <- Predictions
  TimesAll[[kernel]] <- Times
  HyperparamsAll[[kernel]] <- Hyperparams
  SummariesAll[[kernel]] <- summaries
}

# Predictions for the Linear Kernel
head(PredictionsAll$Linear)

#Times of execution for the Linear Kernel
TimesAll$Linear

# Elements of SummariesAll
names(SummariesAll)

# Elements of summaries for the Linear Kernel
names(SummariesAll$Linear)

# Summaries by Line
head(SummariesAll$Linear$line)

#Summaries by Enviroment
SummariesAll$Linear$env[, 1:8]
SummariesAll$Linear$env[, 9:15]
SummariesAll$Linear$env[, 16:19]

#Summaries by Fold
SummariesAll$Linear$fold[, 1:8]
SummariesAll$Linear$fold[, 9:15]
SummariesAll$Linear$fold[, 16:19]

# First rows of Hyperparams
head(HyperparamsAll$Linear)

# Last rows of Hyperparams
tail(HyperparamsAll$Linear)