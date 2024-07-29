setwd("~/Desktop/SCMA EXAM 2/")

# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(e1071)

# Load the dataset
data <- read.csv("bank-additional-full.csv", sep = ";")

# Check the column names
colnames(data)

# Convert the target variable to a factor
data$y <- as.factor(ifelse(data$y == "yes", 1, 0))

# Handle categorical variables by converting them to factors
categorical_vars <- c('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome')
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$y, p = .8, list = FALSE, times = 1)
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

# Logistic Regression model
logistic_model <- glm(y ~ ., data = dataTrain, family = binomial)
summary(logistic_model)

# Predict and evaluate Logistic Regression model
logistic_pred <- predict(logistic_model, newdata = dataTest, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)


# Confusion matrix for Logistic Regression
logistic_conf_matrix <- confusionMatrix(as.factor(logistic_pred_class), dataTest$y)
logistic_conf_matrix

#The results of the Logistic Regression model for predicting bank term deposit subscriptions show a strong overall performance. The model achieves an accuracy of 91.05%, indicating that it correctly predicts whether a client will subscribe to a term deposit in 91.05% of cases. The confidence interval (95% CI) for accuracy ranges from 90.42% to 91.66%, suggesting robust and consistent performance. The model’s precision is high at 92.91%, meaning that when it predicts a client will subscribe, it is usually correct. Sensitivity, or recall, is also excellent at 97.35%, indicating that the model successfully identifies the vast majority of clients who will subscribe. However, the specificity is lower at 41.49%, showing that the model has difficulty correctly identifying clients who will not subscribe. The Kappa statistic of 0.4646 signifies moderate agreement beyond chance. Additionally, the very low p-values for the accuracy comparison with the No Information Rate and Mcnemar’s Test indicate that the model’s performance is significantly better than a naive classifier. Overall, while the model is highly effective in predicting positive cases, there is potential for improvement in correctly identifying negative cases to enhance its overall predictive power.

# AUC-ROC for Logistic Regression
logistic_roc <- roc(dataTest$y, logistic_pred)
plot(logistic_roc, col = "blue")
auc(logistic_roc)

#The ROC curve for the Logistic Regression model illustrates its strong performance in predicting bank term deposit subscriptions. The curve rises sharply towards the top left corner, indicating a high True Positive Rate (Sensitivity) and a low False Positive Rate across various thresholds, suggesting the model effectively distinguishes between clients who will and will not subscribe to a term deposit. The steep ascent and proximity to the top left corner imply a high Area Under the Curve (AUC), close to 1, reflecting excellent discriminatory power. This analysis, combined with previous metrics like accuracy, precision, and sensitivity, confirms the model’s robustness and reliability, making it a valuable tool for the bank in forecasting customer behavior and optimizing marketing efforts.

# Decision Tree model
tree_model <- rpart(y ~ ., data = dataTrain, method = "class")
rpart.plot(tree_model)


#The Decision Tree model for predicting bank term deposit subscriptions is visualized in the provided diagram. The root node splits based on the nr.employed feature, with a threshold of 5088, indicating whether the number of employees is greater than or equal to 5088. The tree then further splits based on duration and poutcome, highlighting the importance of call duration and the outcome of the previous marketing campaign. Nodes at the bottom levels show class predictions (0 or 1) along with probabilities and percentages of samples reaching those nodes. For instance, if nr.employed is less than 5088 and duration is less than 160, the model predicts a higher likelihood (60%) of term deposit subscription. Conversely, the majority of cases with low duration and nr.employed over 5088 are classified as non-subscribers (0). This Decision Tree model reveals that key factors influencing subscription decisions include employment numbers, call duration, and the result of prior marketing efforts, providing valuable insights for targeting potential customers.

# Predict and evaluate Decision Tree model
tree_pred <- predict(tree_model, newdata = dataTest, type = "class")
tree_conf_matrix <- confusionMatrix(tree_pred, dataTest$y)
tree_conf_matrix

#The Decision Tree model for predicting bank term deposit subscriptions achieves an accuracy of 91.21%, with a 95% confidence interval ranging from 90.58% to 91.81%. The model shows a high sensitivity of 96.52%, indicating it effectively identifies clients who will subscribe to a term deposit, though the specificity is lower at 49.35%, reflecting a moderate ability to correctly identify non-subscribers. The precision (Positive Predictive Value) is 93.75%, indicating a high proportion of correctly predicted positive cases. The Kappa statistic is 0.5107, signifying moderate agreement beyond chance. The p-values for both the accuracy comparison to the No Information Rate and McNemar’s Test indicate significant model performance improvements over random guessing. While the balanced accuracy is 72.94%, suggesting overall effectiveness, the model could benefit from improved specificity to better distinguish non-subscribers. Overall, the Decision Tree model is robust in predicting subscriptions but has room for enhancement in accurately identifying non-subscribers.

# AUC-ROC for Decision Tree
tree_pred_prob <- predict(tree_model, newdata = dataTest, type = "prob")[,2]
tree_roc <- roc(dataTest$y, tree_pred_prob)
plot(tree_roc, col = "red")
auc(tree_roc)

#The ROC curve for the Decision Tree model, shown in red, demonstrates the model’s ability to distinguish between clients who will and will not subscribe to a bank term deposit. The curve rises steeply towards the top left corner, indicating a high True Positive Rate (Sensitivity) and a relatively low False Positive Rate. The Area Under the Curve (AUC) is 0.8724, which signifies strong discriminatory power, with values closer to 1 indicating better performance. This AUC value suggests that the Decision Tree model is effective at differentiating between the two classes. However, the curve’s shape also highlights that while sensitivity is high, the model may still face challenges in achieving a perfect balance between sensitivity and specificity. Overall, the Decision Tree model is reliable in predicting customer subscriptions, although there is room for improving specificity to reduce false positives.

# Visualization of Decision Tree structure
rpart.plot(tree_model)

#The visualization of the Decision Tree model for predicting bank term deposit subscriptions shows the hierarchical structure of decisions made based on various features. The root node splits on nr.employed, where values greater than or equal to 5088 lead to a higher probability of predicting a non-subscriber (0). For nodes where nr.employed is less than 5088, the tree further splits on duration, indicating the length of the last call is a significant factor. For example, shorter call durations (less than 160) combined with outcomes of previous campaigns being either ‘failure’ or ‘nonexistent’ increase the likelihood of predicting a subscriber (1). The tree highlights key patterns such as high employment numbers and short call durations correlating with non-subscription, while certain combinations of shorter calls and negative previous campaign outcomes are more likely to result in subscriptions. This structured breakdown provides clear insights into the factors most influential in predicting customer behavior, offering valuable information for optimizing marketing strategies and improving subscription rates.

# Display metrics for Logistic Regression
cat("Logistic Regression Metrics:\n")
cat("Accuracy: ", logistic_conf_matrix$overall['Accuracy'], "\n")
cat("Precision: ", logistic_conf_matrix$byClass['Pos Pred Value'], "\n")
cat("Recall: ", logistic_conf_matrix$byClass['Sensitivity'], "\n")
cat("F1 Score: ", logistic_conf_matrix$byClass['F1'], "\n")
cat("AUC: ", auc(logistic_roc), "\n")

# Display metrics for Decision Tree
cat("Decision Tree Metrics:\n")
cat("Accuracy: ", tree_conf_matrix$overall['Accuracy'], "\n")
cat("Precision: ", tree_conf_matrix$byClass['Pos Pred Value'], "\n")
cat("Recall: ", tree_conf_matrix$byClass['Sensitivity'], "\n")
cat("F1 Score: ", tree_conf_matrix$byClass['F1'], "\n")
cat("AUC: ", auc(tree_roc), "\n")

# Interpretation of Results

# Logistic Regression Coefficients
cat("Logistic Regression Coefficients:\n")
print(summary(logistic_model))
cat("Odds Ratios:\n")
print(exp(coef(logistic_model)))


#The summary of the Logistic Regression model provides detailed insights into the significance and impact of various predictors on the likelihood of subscribing to a term deposit. Key variables include the client’s job type, education level, contact method, and previous campaign outcomes. For instance, ‘jobblue-collar’ and ‘jobservices’ have negative coefficients, indicating that clients in these job categories are less likely to subscribe. In contrast, a positive coefficient for ‘jobretired’ suggests a higher likelihood of subscription. Education levels such as ‘educationilliterate’ show a positive impact, whereas ‘educationhigh.school’ shows a negative impact. The variable ‘contacttelephone’ significantly decreases the probability of subscription, while outcomes like ‘poutcomesuccess’ from previous campaigns significantly increase it.

#The odds ratios further quantify these effects. For example, the odds ratio for ‘jobblue-collar’ is 0.765, indicating that the odds of subscription for blue-collar workers are 23.5% lower compared to the baseline category. Conversely, ‘poutcomesuccess’ has an odds ratio of 2.922, meaning successful outcomes from previous campaigns triple the odds of subscription. Other significant predictors include ‘monthmar’, ‘cons.price.idx’, and ‘euribor3m’, which have substantial impacts on the subscription likelihood. The comprehensive model results underscore the importance of demographic, socio-economic, and campaign-related factors in predicting customer behavior, offering valuable insights for targeted marketing strategies.

# Decision Tree Structure
cat("Decision Tree Structure:\n")
print(tree_model)
cat("Variable Importance:\n")
print(varImp(tree_model))

#The Decision Tree model’s structure and variable importance reveal significant insights into the factors influencing bank term deposit subscriptions. The tree’s primary split is based on nr.employed, with subsequent splits on duration, poutcome, and other factors, indicating these variables’ strong predictive power. The importance rankings show that duration of the last contact is the most influential variable, followed by euribor3m, nr.employed, poutcome, and pdays. The highest importance score, 2182.71633, for duration, highlights the critical role of call duration in predicting customer subscription likelihood. Variables like cons.conf.idx and cons.price.idx also play significant roles, with scores of 101.09398 and 49.69123, respectively, indicating their impact on the model’s predictions. This analysis underscores the importance of employment numbers, call duration, economic indicators, and previous campaign outcomes in forecasting term deposit subscriptions, providing valuable guidance for targeted marketing and customer engagement strategies.

#FINAL INTERPRETATION
# While both models perform well, Logistic Regression slightly edges out the Decision Tree in terms of AUC and recall, making it a better model for this specific task. The higher AUC indicates that Logistic Regression has a better ability to discriminate between the positive and negative classes, and the higher recall suggests it is better at identifying true positives. Therefore, Logistic Regression would be the preferred model.