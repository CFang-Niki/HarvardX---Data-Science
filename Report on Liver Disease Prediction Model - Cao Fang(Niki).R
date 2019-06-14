#Download R packages and the Liver dataset#
library(corrplot)
library(caret)
library(dplyr)
library(dslabs)
library(Rborist)
library(readr)
library(tibble)
Liver <- read_csv("C:/Users/fragr/Documents/indian-liver-patient-records/indian_liver_patient.csv")

#Basic features of the data#
Liver %>% as_tibble()

#Redefine the "Dataset" column#
colnames(Liver)[colnames(Liver)=="Dataset"] <- "Disease"
Liver$Disease[Liver$Disease == 1] <- 0
Liver$Disease[Liver$Disease == 2] <- 1
Liver$Disease <- as.factor(Liver$Disease)

str(Liver$Disease)

#Remove NAs#
colSums(sapply(Liver, is.na))

Liver <- na.omit(Liver)

dim(Liver)

#Descriptive statistic of the Liver Disease dataset#
Liver %>% as_tibble()

#Age Distribution across Groups#
summary(Liver$Age)

Liver %>% ggplot(aes(x = "", y = Age)) +
  geom_boxplot(color = "black", fill = "orange") +
  ylab("Age") +
  ggtitle("Age Distribution")

Liver %>% ggplot(aes(x = Gender, y = Age)) +
  geom_boxplot(aes(fill = Gender), color = "black") +
  ylab("Age") +
  ggtitle("Age Distribution across Gneder Group")

Liver %>% ggplot(aes(x = Disease, y = Age)) +
  geom_boxplot(aes(fill = Disease), color = "black") +
  ylab("Age") +
  ggtitle("Age Distribution across Disease Group")

#Gender Distribution across Groups#
summary(Liver$Gender) 

Liver %>% ggplot(aes(Gender)) + 
  geom_bar(aes(fill = Gender), color = "black") + 
  ggtitle("Gender Counts") + 
  theme(legend.position="none")

Liver %>% ggplot(aes(Disease, Age)) + 
  geom_boxplot(aes(fill = Gender), color = "black") +
  ggtitle("Gender Distribution across Disease/Age Group")

#Total_Bilirubin Distribution across Groups#
summary(Liver$Total_Bilirubin)

Liver %>% ggplot(aes(Age, Total_Bilirubin)) + 
  geom_point(aes(color = Gender)) +
  ggtitle("Total_Bilirubin Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Direct_Bilirubin Distribution across Groups#
summary(Liver$Direct_Bilirubin)

Liver %>% ggplot(aes(Age, Direct_Bilirubin)) + 
  geom_point(aes(color = Gender)) +
  ggtitle("Direct_Bilirubin Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Alkaline_Phosphotase Distribution across Groups#
summary(Liver$Alkaline_Phosphotase)

Liver %>% ggplot(aes(Age, Alkaline_Phosphotase)) + 
  geom_point(aes(color = Gender)) +
  ggtitle("Alkaline_Phosphotase Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Alamine_Aminotransferase Distribution across Groups#
summary(Liver$Alamine_Aminotransferase)

Liver %>% ggplot(aes(Age, Alamine_Aminotransferase)) + 
  geom_point(aes(color = Gender)) +
  ggtitle("Alamine_Aminotransferase Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Aspartate_Aminotransferase Distribution across Groups#
summary(Liver$Aspartate_Aminotransferase)

Liver %>% ggplot(aes(Age, Aspartate_Aminotransferase)) + 
  geom_point(aes(color = Gender)) +
  ggtitle("Aspartate_Aminotransferase Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Total_Protiens Distribution across Groups#
summary(Liver$Total_Protiens)

Liver %>% ggplot(aes(Age, Total_Protiens)) + 
  geom_boxplot(aes(fill = Gender), color = "black") +
  ggtitle("Total_Protiens Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Albumin Distribution across Groups#
summary(Liver$Albumin)

Liver %>% ggplot(aes(Age, Albumin)) + 
  geom_boxplot(aes(fill = Gender), color = "black") +
  ggtitle("Albumin Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Albumin_and_Globulin_Ratio Distribution across Groups#
summary(Liver$Albumin_and_Globulin_Ratio)

Liver %>% ggplot(aes(Age, Albumin_and_Globulin_Ratio)) + 
  geom_boxplot(aes(fill = Gender), color = "black") +
  ggtitle("Albumin_and_Globulin_Ratio Distribution across Disease/Gender Group") +
  facet_grid(Gender~Disease)

#Building the prediction model#
#Remove the highly correlated predictors#
non_numeric_cols <- c('Gender', 'Disease')
correlations <- cor(Liver[, !(names(Liver) %in% non_numeric_cols)])
correlations

corrplot(correlations, tl.cex = 0.6, tl.col = "blue")

high_Cor <- findCorrelation(correlations, cutoff = 0.5, names=TRUE)
high_Cor

Liver <- Liver[, !(names(Liver) %in% high_Cor)]

dim(Liver)

#Create train set and test set#
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(Liver$Disease, times = 1, p = 0.5, list = FALSE)
train_set <- Liver[-test_index,]
test_set <- Liver[test_index,]

#Model 1 - Generalized Linear Model#
set.seed(1, sample.kind = "Rounding")
fit.glm <- train(Disease ~ ., data = train_set, method = "glm", family = "binomial")
y_hat_glm <- predict(fit.glm, test_set)
confusionMatrix(y_hat_glm, test_set$Disease)

glm_accuracy <- confusionMatrix(y_hat_glm, test_set$Disease)$overall["Accuracy"]
glm_sensitivity <- sensitivity(y_hat_glm, test_set$Disease)
glm_specificity <- specificity(y_hat_glm, test_set$Disease)

#Model 2 - KNN Model#
set.seed(1, sample.kind = "Rounding")
fit.knn <- train(Disease ~ ., data = train_set, method = "knn", tuneGrid = data.frame(k = 5))
y_hat_knn <- predict(fit.knn, test_set, type = "raw")
confusionMatrix(y_hat_knn, test_set$Disease)

#Tuning Parameter k#
set.seed(1, sample.kind = "Rounding")
fit.knn_k <- train(Disease ~ ., data = train_set, method = "knn", tuneGrid = data.frame(k = seq(1,100,1)))
fit.knn_k$bestTune

#Best k#
set.seed(1, sample.kind = "Rounding")
fit.knn_bk <- train(Disease ~ ., data = train_set, method = "knn", tuneGrid = data.frame(k = 62))
y_hat_knn_bk <- predict(fit.knn_bk, test_set, type = "raw")
confusionMatrix(y_hat_knn_bk, test_set$Disease)

knn_accuracy <- confusionMatrix(y_hat_knn_bk, test_set$Disease)$overall["Accuracy"]
knn_sensitivity <- sensitivity(y_hat_knn_bk, test_set$Disease)
knn_specificity <- specificity(y_hat_knn_bk, test_set$Disease)

#Model 3 - Linear Discriminant Analysis Model#
set.seed(1, sample.kind = "Rounding")
fit.lda <- train(Disease ~ ., data = train_set, method = "lda")
y_hat_lda <- predict(fit.lda, test_set)
confusionMatrix(y_hat_lda, test_set$Disease)

lda_accuracy <- confusionMatrix(y_hat_lda, test_set$Disease)$overall["Accuracy"]
lda_sensitivity <- sensitivity(y_hat_lda, test_set$Disease)
lda_specificity <- specificity(y_hat_lda, test_set$Disease)

#Model 4 -  Random Forest Model 1#
set.seed(1, sample.kind = "Rounding")
fit.rf <- train(Disease ~ ., data = train_set, method = "rf", tuneGrid = expand.grid(.mtry = 5), 
                trControl = trainControl(method = "cv", number = 10, search = "grid"),importance = TRUE, 
                nodesize = 50, maxnodes = 25, ntree = 1000, prox = TRUE)
y_hat_rf <- predict(fit.rf, test_set)
confusionMatrix(y_hat_rf, test_set$Disease)

#Tunning Parameter mtry#
set.seed(1, sample.kind = "Rounding")
tuneGrid <- expand.grid(.mtry = seq(1,10,1))
fit.rf_m <- train(Disease ~ ., data = train_set, method = "rf", tuneGrid = tuneGrid, 
                trControl = trainControl(method = "cv", number = 10, search = "grid"),importance = TRUE, nodesize = 50, 
                maxnodes = 25, ntree = 1000, prox = TRUE)
fit.rf_m

#Best mtry#
set.seed(1, sample.kind = "Rounding")
fit.rf_bm <- train(Disease ~ ., data = train_set, method = "rf", tuneGrid = expand.grid(.mtry = 2), 
                trControl = trainControl(method = "cv", number = 10, search = "grid"),importance = TRUE, nodesize = 50, 
                maxnodes = 25, ntree = 1000, prox = TRUE)
y_hat_rf_bm <- predict(fit.rf_bm, test_set)
confusionMatrix(y_hat_rf_bm, test_set$Disease)

rf_accuracy <- confusionMatrix(y_hat_rf_bm, test_set$Disease)$overall["Accuracy"]
rf_sensitivity <- sensitivity(y_hat_rf_bm, test_set$Disease)
rf_specificity <- specificity(y_hat_rf_bm, test_set$Disease)

#Model 5 - Random Forest Model 2#
set.seed(1, sample.kind = "Rounding")
fit.rb <- train(Disease ~ ., method = "Rborist", tuneGrid = data.frame(predFixed = 2, minNode = 10), data = train_set)
y_hat_rb <- predict(fit.rb, test_set)
confusionMatrix(y_hat_rb, test_set$Disease)

#Tunning Parameter predFixed & minNode#
set.seed(1, sample.kind = "Rounding")
predFixed <- seq(1,10,1)
minNode <- seq(5, 300, 10)
fit.rb_pm <- train(Disease ~ ., method = "Rborist", tuneGrid = data.frame(predFixed = predFixed, minNode = minNode), data = train_set)
fit.rb_pm

#Best predFixed & minNode#
set.seed(1, sample.kind = "Rounding")
fit.rb_bpm <- train(Disease ~ ., method = "Rborist", tuneGrid = data.frame(predFixed = 1, minNode = 5), data = train_set)
y_hat_rb_bpm <- predict(fit.rb_bpm, test_set)
confusionMatrix(y_hat_rb_bpm, test_set$Disease)

rb_accuracy <- confusionMatrix(y_hat_rb_bpm, test_set$Disease)$overall["Accuracy"]
rb_sensitivity <- sensitivity(y_hat_rb_bpm, test_set$Disease)
rb_specificity <- specificity(y_hat_rb_bpm, test_set$Disease)

#Results#
Results <- data.frame(Method = "Generalized Linear Model", Accuracy = glm_accuracy, Sensitivity = glm_sensitivity, specificity = glm_specificity) %>%
  bind_rows(data_frame(Method = "KNN Model", Accuracy = knn_accuracy, Sensitivity = knn_sensitivity, specificity = knn_specificity)) %>%
  bind_rows(data_frame(Method = "Linear Discriminant Analysis Model", Accuracy = lda_accuracy, Sensitivity = lda_sensitivity, specificity = lda_specificity)) %>%
  bind_rows(data_frame(Method = "Random Forest Model 1", Accuracy = rf_accuracy, Sensitivity = rf_sensitivity, specificity = rf_specificity)) %>% 
  bind_rows(data_frame(Method = "Random Forest Model 2", Accuracy = rb_accuracy, Sensitivity = rb_sensitivity, specificity = rb_specificity))
print(Results)

#Github Link to the project#
Github link: https://github.com/CFang-Niki/HarvardX---Data-Science.git