rm(list = ls())
setwd("~/Desktop/NAB Datathon")
#---------------------------------------------------------#
library(pacman)
pacman::p_load(devtools,
               blogdown,
               readr,
               openxlsx,
               rvest,
               magrittr,
               
               tools,
               scales,
               stringr,
               imputeTS,
               lubridate,
               xts,
               zoo,
               
               tidyr,
               data.table,
               dplyr,
               officer,
               flextable,
               flexdashboard,
               
               ggplot2,
               ggthemes,
               ggmap,
               ggrepel,
               plotly,
               networkD3,
               ggalluvial,
               cowplot,
               grid,
               gridExtra,
               rgdal,
               rgeos,
               leaflet,
               leaflet.extras,
               shiny,
               shinyjs,
               shinydashboard,
               
               lattice,
               waterfall,
               forcats,
               rvg,
               BiocManager,
               
               UBL,  # Balancing Biased class in a sample
               MASS, # For Linear and Quadratic discriminant Analysis Model
               mda,  # For Mixture and Flexible discriminant Analysis Model
               #klaR, # For Regularized discriminant Analysis
               IC2,
               ROCR,
               #caret,
               #mlr,
               #mlr3,
               party,
               randomForest,
               Hmisc,
               ParamHelpers,install = TRUE)

'%!in%' <- Negate('%in%')
#---------------------------------------------------------#
# QoL <- read.csv(file = "QoL.csv",stringsAsFactors = F)
# QoL.train <- read.csv(file = "QoL-train.csv",stringsAsFactors = F)

bnk.fraud <- read.csv(file = "creditcard_sample.csv",stringsAsFactors = F)

#------------------------- Data Exploration --------------------------------#
colSums(is.na(bnk.fraud)) %>% View()
rowSums(is.na(bnk.fraud)) %>% View()

summary(bnk.fraud$Amount)
histogram(bnk.fraud$Amount)
boxplot(bnk.fraud$Amount)

table(bnk.fraud$Class)
prop.table(table(bnk.fraud$Class))

res <- cor(bnk.fraud)
round(res, digits = 2)

#------------------------- Data Processing ---------------------------------#

subset.fraud.df <- bnk.fraud %>% dplyr::select(V1:V15,Amount,Class) %>%
  mutate(Amount = log10(Amount),
         Amount = if_else(is.finite(Amount),Amount,0),
         Class = if_else(Class == 0,"FALSE","TRUE"),
         Class = factor(Class))

par(mfrow = c(3,3))
density(subset.fraud.df$V1) %>% plot(main = "PC 1 Distribution")
density(subset.fraud.df$V2) %>% plot(main = "PC 2 Distribution")
density(subset.fraud.df$V3) %>% plot(main = "PC 3 Distribution")
density(subset.fraud.df$V4) %>% plot(main = "PC 4 Distribution")
density(subset.fraud.df$V5) %>% plot(main = "PC 5 Distribution")
density(subset.fraud.df$V6) %>% plot(main = "PC 6 Distribution")
density(subset.fraud.df$V7) %>% plot(main = "PC 7 Distribution")
density(subset.fraud.df$V8) %>% plot(main = "PC 8 Distribution")
density(subset.fraud.df$V9) %>% plot(main = "PC 9 Distribution") # Normal

density(subset.fraud.df$V10) %>% plot(main = "PC 10 Distribution") # Normal
density(subset.fraud.df$V11) %>% plot(main = "PC 11 Distribution")
density(subset.fraud.df$V12) %>% plot(main = "PC 12 Distribution")
density(subset.fraud.df$V13) %>% plot(main = "PC 13 Distribution") # Normal
density(subset.fraud.df$V14) %>% plot(main = "PC 14 Distribution")
density(subset.fraud.df$V15) %>% plot(main = "PC 15 Distribution")

#--------------------------------- Sampling --------------------------------#
# Random sampling index
train_index <- sample(1:nrow(subset.fraud.df), 0.75 * nrow(subset.fraud.df))
test_index <- setdiff(1:nrow(subset.fraud.df),train_index)

# Sample for train and test data
train.df <- subset.fraud.df[train_index, -19]
table(train.df$Class)
prop.table(table(train.df$Class))

# Balance Biased Class dataset
# train.df.ISA <- UBL::ImpSampClassif(Class~.,train.df,C.perc = list('FALSE' = 1,
#                                                                    'TRUE' = 200))

train.df.ISA <- UBL::ImpSampClassif(Class~.,train.df,C.perc = list('FALSE' = 1,
                                                                   'TRUE' = 410))

table(train.df.ISA$Class)
prop.table(table(train.df.ISA$Class))

write.xlsx(train.df.ISA, file = "train.df.xlsx")

test.df <- subset.fraud.df[test_index, -19]
table(test.df$Class)
prop.table(table(test.df$Class))

#----------------------------- Model Building ------------------------------#

#----------------- Logistic Regression ---------------#
model.glm <- glm(Class ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15 + Amount,
                 data = train.df.ISA,
                 family = binomial)

summary(model.glm)

# model.glm.1 <- glm(Class ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + Amount,
#                    data = train.df.ISA,
#                    family = binomial)
# 
# summary(model.glm.1)
# 
# model.glm.2 <- glm(Class ~ V1 + V2 + V3 + V4 + V5 + V6 + V8 + V9 + V10 + Amount,
#                    data = train.df.ISA,
#                    family = binomial)
# 
# summary(model.glm.2)

glm.pred <- predict(model.glm,
                    test.df,
                    type = "response")

# Confusion Matrix
table(test.df$Class,glm.pred > 0.5)

# table(test.df$Class, ifelse(glm.pred > 0.5, "TRUE","FALSE"))

# Accuracy Rate
mean(ifelse(glm.pred > 0.5, "TRUE","FALSE") == test.df$Class) # 99%

# Error Rate
mean(ifelse(glm.pred > 0.5, "TRUE","FALSE") != test.df$Class) # 1%

# AUC
ROCR::prediction(glm.pred, test.df$Class) %>%
  ROCR::performance(measure = "auc") %>%
  .@y.values

# ROC Curve
ROC.glm <- ROCR::prediction(glm.pred, test.df$Class) %>%
  ROCR::performance(measure = "tpr", x.measure = "fpr")

#----------------- QDA ---------------#

model.qda <- MASS::qda(Class ~ V1 + V2 + V4 + V5 + V6 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15 + Amount,
                       data = train.df.ISA)

qda.pred <- predict(model.qda, test.df)

sum(qda.pred$posterior[,1] >= .5) # No. of Fraud Prediction
sum(qda.pred$posterior[,2] >= .5) # No. of Non-Fraud Prediction

# Confusion Matrix GLM vs QDA
list(table(test.df$Class,glm.pred > 0.5),
     table(test.df$Class, qda.pred$class)
)

# Recall Of QDA and GLM
#--- QDA: 0.19%
#--- GLM: 19%

# Precision Of QDA and GLM
#--- QDA: 92%
#--- GLM: 91%

# Accuracy Rate
mean(test.df$Class == qda.pred$class) # 96%

# Error Rate
mean(test.df$Class != qda.pred$class) # ~4%

# AUC
ROCR::prediction(qda.pred$posterior[,2], test.df$Class) %>%
  ROCR::performance(measure = "auc") %>%
  .@y.values

# ROC Curve
ROC.qda <- ROCR::prediction(qda.pred$posterior[,2], test.df$Class) %>%
  ROCR::performance(measure = "tpr", x.measure = "fpr")

# ROC Curve GLM vs QDA
ROCR::plot(ROC.glm, col = "Red")
ROCR::plot(ROC.qda, add = TRUE, col = "Blue")

#--------------------- Prediction in Complete Data using QDA ----------------------#

set.seed(3137924)

creditCard.full.df <- read.csv(file = "creditcard.csv",stringsAsFactors = F)

creditCard.full.df <- creditCard.full.df %>% dplyr::select(V1,V2,V4:V6,V8:V15,Amount,Class) %>%
  mutate(Amount = log10(Amount),
         Amount = if_else(is.finite(Amount),Amount,0),
         Class = if_else(Class == 0,"FALSE","TRUE"),
         Class = factor(Class))

samp.size = floor(0.8 * nrow(creditCard.full.df))
train_ind = sample(seq_len(nrow(creditCard.full.df)), size = samp.size)

train = creditCard.full.df[train_ind,]
test = creditCard.full.df[-train_ind,]

identical(subset.fraud.df[-c(1)],train)
identical(subset.fraud.df[-c(1)],test)
identical(test.df[-c(1)],test)

qda.pred.full = predict(model.qda,test)

table(test$Class,qda.pred.full$class)
