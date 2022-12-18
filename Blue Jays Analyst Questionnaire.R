library(tidyverse)
library(e1071)
library(caret)
library(randomForestSRC)
library(gbm)
library(pROC)

training <- read_csv("training.csv")
deploy <- read_csv("deploy.csv")

training <- training %>% mutate(SpinRate = as.numeric(SpinRate),
                                InPlay = factor(InPlay, levels = c(0,1), labels = c("No","Yes")),
                                across(Velo:InducedVertBreak, scale)) %>%
  filter(!is.na(SpinRate))

training <- as.data.frame(training)

deploy_cleaned <- deploy %>% mutate(SpinRate = as.numeric(SpinRate),
                                across(Velo:InducedVertBreak, scale)) 
deploy_cleaned <- as.data.frame(deploy_cleaned)

training %>% group_by(InPlay) %>%
  tally()

#73% correct if just predicted all outcomes were No

#Create training and test sets via stratified random sampling
set.seed(123)
split1 <- createDataPartition(training$InPlay, p = .7)[[1]]
test <- training[-split1,]
training <- training[ split1,]
set.seed(123)
split2 <- createDataPartition(test$InPlay, p = 1/3)[[1]]
evaluation <- test[ split2,]
test <- test[-split2,]

evaluate_cutoff <- evaluation

# Random Forest -----------------------------------------------------------

ctrl <- trainControl(method = "cv", classProbs = TRUE)

set.seed(1410)
rfFit <- train(InPlay ~ ., data = training,
               method = "rf",
               trControl = ctrl,
               ntree = 1500,
               tuneLength = 5,
               metric = "ROC")

evaluate_cutoff$RF <- predict(rfFit,
                          newdata = evaluation,
                          type = "prob")[,1]

rfROC <- roc(evaluate_cutoff$InPlay, evaluate_cutoff$RF,
             levels = rev(levels(evaluate_cutoff$InPlay)))

plot(rfROC, legacy.axes = TRUE) 

rfThresh <- coords(rfROC, x = "best", best.method = "closest.topleft") 
rfThresh


# Boosting ----------------------------------------------------------------

set.seed(1)
caretGrid <- expand.grid(interaction.depth=c(1,3,5), n.trees = c(100,200,300,500,1000,2000,5000),
                         shrinkage=c(0.1,0.01, 0.001),n.minobsinnode = 10)

gbm.caret <- train(InPlay ~ ., data = training, 
                   distribution="bernoulli", 
                   method="gbm",
                   trControl=ctrl, 
                   verbose=FALSE, 
                   tuneGrid=caretGrid, 
                   metric="ROC") 

evaluate_cutoff$boost <- predict(gbm.caret,
                              newdata = evaluation,
                              type = "prob")[,1]

boostROC <- roc(evaluate_cutoff$InPlay, evaluate_cutoff$boost,
             levels = rev(levels(evaluate_cutoff$InPlay)))

plot(boostROC, legacy.axes = TRUE) 

boostThresh <- coords(boostROC, x = "best", best.method = "closest.topleft") 
boostThresh

# KNN ---------------------------------------------------------------------

set.seed(1)

knn_fit <- train(InPlay ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:100),
             trControl  = ctrl,
             metric     = "ROC",
             data       = training)

evaluate_cutoff$knn <- predict(knn_fit,
                                 newdata = evaluation,
                                 type = "prob")[,1]

knnROC <- roc(evaluate_cutoff$InPlay, evaluate_cutoff$knn,
                levels = rev(levels(evaluate_cutoff$InPlay)))

plot(knnROC, legacy.axes = TRUE) 

knnThresh <- coords(knnROC, x = "best", best.method = "closest.topleft") 
knnThresh

# Logistic Regression -----------------------------------------------------

set.seed(1)

glm.fits <- glm(InPlay ~ Velo + HorzBreak + InducedVertBreak + HorzBreak*InducedVertBreak,
                data = training, family = binomial)

summary(glm.fits)

evaluate_cutoff$log <- predict(glm.fits,
                               newdata = evaluation,
                               type = "response")

logROC <- roc(evaluate_cutoff$InPlay, evaluate_cutoff$log,
              levels = rev(levels(evaluate_cutoff$InPlay)))

plot(logROC, legacy.axes = TRUE) 

logThresh <- coords(logROC, x = "best", best.method = "closest.topleft") 
logThresh


# Predict On Test Data to decide best model -------------------------------

test_predictions <- test

test_predictions$rf <- predict(rfFit,newdata = test,type = "prob")[,1]
roc(test_predictions$InPlay, test_predictions$rf,
              levels = rev(levels(test_predictions$InPlay)))

test_predictions$boost <- predict(gbm.caret,newdata = test,type = "prob")[,1]
roc(test_predictions$InPlay, test_predictions$boost,
    levels = rev(levels(test_predictions$InPlay)))

test_predictions$knn <- predict(knn_fit,newdata = test,type = "prob")[,1]
roc(test_predictions$InPlay, test_predictions$knn,
    levels = rev(levels(test_predictions$InPlay)))

test_predictions$log <- predict(glm.fits,newdata = test,type = "response")
roc(test_predictions$InPlay, test_predictions$log,
    levels = rev(levels(test_predictions$InPlay)))

test_predictions <- test_predictions %>% mutate(prediction = ifelse(log > logThresh$threshold, "Yes", "No"),
                                                prediction = as.factor(prediction))

confusionMatrix(data = test_predictions$prediction,
                reference = test_predictions$InPlay,
                positive = "Yes")

# Create Assignments for Deploy Data --------------------------------------

deploy_log_odds <- predict(glm.fits, deploy_cleaned, type = "response")

deploy_final <- deploy %>% mutate(log_odds = deploy_log_odds,
                                  InPlay = ifelse(log_odds > logThresh$threshold,"Yes","No")) %>%
  select(-log_odds)

write_excel_csv(deploy_final,"deploy_predictions.csv",na = "")


# Plot Variables ----------------------------------------------------------

training <- read_csv("training.csv")

training_long <- training %>% mutate(SpinRate = as.numeric(SpinRate),
                                InPlay = factor(InPlay, levels = c(0,1), labels = c("No","Yes"))) %>%
  filter(!is.na(SpinRate)) %>%
  pivot_longer(cols = c(Velo:InducedVertBreak), names_to = "variable", values_to = "value")

training_long %>% ggplot(aes(x = InPlay, y = value)) + geom_boxplot() +
  facet_wrap(~variable, scales = "free") +
  theme_classic()

ggsave("predictor_plot.png")

