# OPAN 6603 - Project 1 ####

# Mike Johnson | Andrew Singh

## Set up ####

# Libraries
library(tidyverse) # For all things data manipulation
library(GGally) # for data exploration
library(caret) # For confusionMatrix(), training ML models, and more
library(class) # For knn()
library(dplyr) # For some data manipulation and ggplot
library(pROC)  # For ROC curve and estimating the area under the ROC curve
library(fastDummies) # To create dummy variable (one hot encoding)

# Set random seed for reproducibility
set.seed(206)

# Set viz theme
theme_set(theme_classic())

## Load Data ####
df = read_csv("data-raw/EmployeeData.csv")

# Data structure
str(df)

# Update data types
df = 
  df %>% 
  mutate(
    # Dependent Variable
    Attrition = factor(Attrition),
    
    # Predictors
    BusinessTravel = factor(BusinessTravel),
    Education = factor(Education, levels = 1:5, labels = c("Below College", "College", "Bachelor", "Master", "Doctor")),
    Gender = factor(Gender), 
    JobLevel = factor(JobLevel),
    MaritalStatus = factor(MaritalStatus),
    NumCompaniesWorked = as.numeric(NumCompaniesWorked),
    TotalWorkingYears = as.numeric(TotalWorkingYears), 
    EnvironmentSatisfaction = factor(EnvironmentSatisfaction, levels = 1:4, labels = c("Low", "Medium", "High", "Very High")), 
    JobSatisfaction = factor(JobSatisfaction, levels = 1:4, labels = c("Low", "Medium", "High", "Very High"))) 

# Remove Irrelevant Columns
df = 
  df %>% 
  select(
    -EmployeeID,
    -StandardHours)

# Check for NA's
na_summary = df %>% 
  summarise_all(~ sum(is.na(.))) %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "na_count") %>% 
  filter(na_count > 0)

# How should we handle NAs?
na_summary

# Drop NA values
df = na.omit(df)

# Create dummy variables for categorical variables and remove the columns used to create dummy variables
df =
  df %>% 
  dummy_cols(
    select_columns = c('BusinessTravel',
                       'Education', 
                       'Gender', 
                       'JobLevel', 
                       'MaritalStatus',
                       'EnvironmentSatisfaction',
                       'JobSatisfaction'),
    remove_selected_columns = F,
    remove_first_dummy = F
  )

## kNN Model ####

### Step 1: Create a train/test split ####

# Divide 30% of data to test set
test_indices = createDataPartition(1:nrow(df),
                                   times = 1,
                                   p = 0.3)

# Create training set
df_train = df[-test_indices[[1]], ]

# Create test set
df_test = df[test_indices[[1]], ]


### Step 2: Data Exploration ####

# Summary of training set
summary(df_train)

#df_train %>% 
# ggpairs(aes(color = Attrition, alpha = 0.4))

# Viz of attrition distribution
# Imbalanced classes. Will need to downsample.
df_train %>% 
  ggplot(aes(x = Attrition)) + 
  geom_bar(fill = "steelblue") +
  labs(title = "Attrition Distribution")

# Viz of relationship between Education and Attrition
df_train %>% 
  ggplot(aes(x = Gender, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Gender Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))


# Viz of relationship between Age and Attrition
df_train %>% 
  ggplot(aes(x = Age, fill = Attrition)) +
  geom_histogram(binwidth = 10, position = "dodge") +
  facet_grid(~Attrition) +
  labs(title = "Age Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

df_train %>% 
  mutate(age_t = log(Age)) %>% 
  ggplot(aes(x = age_t, fill = Attrition)) +
  geom_histogram() +
  facet_grid(~Attrition) +
  labs(title = "Age Transformed Distribution by Attrition")
  

# Viz of relationship between Education and Attrition
df_train %>% 
  ggplot(aes(x = Education, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Education Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Viz of relationship between Job Satisfaction and Attrition
df_train %>% 
  ggplot(aes(x = JobSatisfaction, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Job Satisfaction Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

# Viz of relationship between Working Years and Attrition
df_train %>% 
  ggplot(aes(x = TotalWorkingYears, fill = Attrition)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  facet_grid(~Attrition) +
  labs(title = "Working Years Distribution by Attrition") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "red"))

df_train %>% 
  mutate(workingyears_t = log(TotalWorkingYears)) %>% 
  ggplot(aes(x = workingyears_t, fill = Attrition)) +
  geom_histogram() +
  facet_grid(~Attrition) +
  labs(title = "Working Years Transformed Distribution by Attrition")

# Viz of distribution of age and marital status

df_train %>% 
  ggplot(aes(x = Age)) +
  geom_histogram(binwidth = 10, position = "dodge") +
  facet_grid(Attrition ~ MaritalStatus) +
  labs(title = "Age Distribution of Employees by Marital Status and Attrition")


df_train %>% 
  filter(Attrition == "Yes") %>% 
  ggplot(aes(x = Age)) +
  geom_histogram(binwidth = 10) +
  facet_grid(~MaritalStatus) +
  labs(title = "Age Distribution of Employees by Marital Status",
       subtitle = "Attritioned Employees") 

### Step 3: Data pre-processing ####

# Downsampling
downsample_df = downSample(x = df_train[ , colnames(df_train) != "Attrition"],
                           y = df_train$Attrition)

colnames(downsample_df)[ncol(downsample_df)] = "Attrition"

downsample_df %>% 
  ggplot(aes(x = Attrition)) + 
  geom_bar(fill = "steelblue") +
  labs(title = "Attrition Distribution")

### Step 4: Feature Engineering ####

downsample_df %>% 
    select(any_of("Attrition"), where(is.numeric))

# Remove factor variables
downsample_df = 
  downsample_df %>% 
  select(any_of("Attrition"), where(is.numeric))

downsample_df %>% 
  select(function(x) ! is.integer(x) & ! is.factor(x))

# Scale predictors due to distance functions being sensitive to scale.

standardizer = 
  preProcess(downsample_df %>% 
               select(function(x) ! is.integer(x) & ! is.factor(x)),
             method = c("center", "scale")
             )

downsample_df = predict(standardizer, downsample_df)

df_test = predict(standardizer, df_test)

### Step 5: Feature & Model Selection ####

knn_classifier = 
  train(
    Attrition ~ .,
    data = downsample_df, # Remove factors
    method = "knn",
    tuneGrid = expand.grid(k = seq(2, 40)),
    trControl = trainControl(method = "cv",
                             number = 10, # 10-fold cross validation
                             classProbs = TRUE, # Enable probability predictions
                             summaryFunction = twoClassSummary), # use twoClassSummary to compute AUC
    metric = "ROC" # ROC give us AUC & silences warning about Accuracy
  )

plot(knn_classifier)

knn_classifier$bestTune

knn_classifier$results

knn_classifier$resample

# Identify variables of importance
knn_varImp = varImp(knn_classifier)

knn_varImp %>% 
  ggplot(aes())

varImp(svm_poly)

### Step 6: Model Validation ####
# Validation completed during training through

### Step 7: Predictions and Conclusions ####

# ROC and AUC
roc_knn =
  roc(df_test$Attrition,
      predict(knn_classifier,
              df_test,
              type = "prob")[["Yes"]]
      )

plot(roc_knn, main = "k Nearest Neighbors")

roc_knn$auc

# Confusion matrix stats
confusion_knn =
  confusionMatrix(data = df_test$Attrition,
                  reference = predict(knn_classifier, df_test, type = "raw"),
                  positive = "Yes"
                  )

confusion_knn$table

# High precision, low recall.
confusion_knn$byClass[c("Precision", "Recall")]


## SVM Model ####

### Step 1: Create a train/test split ####
# Completed in kNN Model

### Step 2: Data Exploration ####
# Completed in kNN Model

### Step 3: Data pre-processing ####
# Completed in kNN Model

### Step 4: Feature Engineering ####
# Completed in kNN Model

### Step 5: Feature & Model Selection ####

# Create trainControl for re-usability
tr_control = trainControl( # store since we will reuse
  method = "cv", number = 10, # 10-fold cross validation
  classProbs = TRUE,  # Enable probability predictions
  summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
)

# SVM Linear
svm_linear = train(
  Attrition ~ .,
  data = downsample_df,
  method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.01, 0.1, 1, 5, 10)),
  trControl = tr_control,
  metric = "ROC",
  na.action = na.pass
)

plot(svm_linear)

svm_linear$bestTune

# SVM Radial
svm_radial = train(
  Attrition ~.,
  data = downsample_df,
  method = "svmRadial",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10), sigma = c(0.5,1,2,3)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_radial)

svm_radial$bestTune

# SVM Poly
svm_poly = train(
  Attrition ~.,
  data = downsample_df,
  method = "svmPoly",
  tuneLength = 4, # will automatically try different parameters with CV
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_poly)

svm_poly$bestTune

### Step 6: Model Validation ####

# Create Validation Table
validation_table =
  bind_rows(
    list(
      svm_linear$resample %>% 
        mutate(
          type = "linear",
          mean_auc = mean(ROC)
        ) %>% 
        cbind(svm_radial$bestTune),
      svm_radial$resample %>% 
        mutate(
          type = "radial",
          mean_auc = mean(ROC)
        ) %>% 
        cbind(svm_radial$bestTune),
      svm_poly$resample %>% 
        mutate(
          type = "polynomial",
          mean_auc = mean(ROC)
        ) %>% 
        cbind(svm_poly$bestTune)
    )
  )

# Visualize model validation
validation_table %>% 
  ggplot(aes(x = ROC)) +
  geom_density(aes(fill = type), alpha = 0.5) + 
  geom_vline(aes(xintercept = mean_auc))+
  facet_wrap(~type)

### Step 7: Predictions and Conclusions ####

# let's choose the best model, polynomial

# get probabilistic predictions on your test set on your chosen model
preds = predict(svm_radial, df_test, type = "prob")

# plot ROC and calculate AUC
roc_radial = roc(
  df_test$Attrition,
  preds$Yes
)

plot(roc_radial, main = "Support Vector Machines")

roc_radial$auc

# pick threshold with highest average of sensitivity and specificity
thresh = roc_radial$thresholds[which.max(roc_radial$sensitivities + roc_radial$specificities)]

# Create Confusion Matrix
confusion_svm = confusionMatrix(
  df_test$Attrition |> factor(),
  ifelse(preds$Yes >= thresh, "Yes", "No") |> factor()
)

confusion_svm$table

confusion_svm$byClass[c("Precision", "Recall")]
