library(xgboost)
library(stringr)
library(dplyr)
library(ggplot2)
library(Matrix)
library(caret)
library(dummies)
library(InformationValue)
library(jiebaR)

#整理training set
df_train_1$Order_Date <- as.numeric(df_train_1$Order_Date)
df_train_1$people_amount <- as.numeric(df_train_1$people_amount)
df_train_1$Begin_Date <- as.numeric(df_train_1$Begin_Date)
df_train_1$days <- as.numeric(df_train_1$days)
df_train_1$price <- as.numeric(df_train_1$price)
df_train_1$PreDays <- as.numeric(df_train_1$PreDays)
df_train_1$preMinutes.去程 <- as.numeric(df_train_1$preMinutes.去程)
df_train_1$preMinutes.回程 <- as.numeric(df_train_1$preMinutes.回程)
df_train_1$Total_Amount_People <- as.numeric(df_train_1$Total_Amount_People)
one_hot_features=c(Unit_name, Source_1, Source_2, SubLine, Total_Amount_People, Amount_Order)
for (var in one_hot_features){
  df_train_1[,var] <- as.numeric(df_train_1[,var]) 
}
#解釋變數只吃0開始
x <- as.factor(df_train_1$deal_or_not)
levels(x) <- 1:length(levels(x))
label <- as.numeric(x) - 1

df_train_1 <- xgb.DMatrix(data = df_train_1 %>% select(-c(order_id,group_id,deal_or_not,product_name,src_dst.去程,src_dst.回程)) %>% as.matrix(),
                          label = label) 
# set random seed
set.seed(666)
gblinear
gbtree
# xgboost 參數設定 (xgboost parameters setup)
params <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eval_metric = "auc",
               #num_class = 2, 
               #nthread = 15,
               eta=1, 
               gamma=0, 
               max_depth=6, 
               min_child_weight=1, 
               subsample=0.7, 
               colsample_bytree=0.7
               #,lambda = 2
)
xgbcv <- xgb.cv(params = params, 
                data = df_train_1, 
                nrounds = 30, 
                nfold = 10, 
                showsd = T, 
                stratified = T, 
                #print_every_n = 10, 
                #early_stopping_rounds = 100, 
                maximize = F, 
                verbose = 1, 
                prediction = TRUE)
# build the model
bst <- xgboost(params = params, 
               data = df_train_1, 
               nround = 150, 
               verbose = 1)
importance <- xgb.importance( model = bst)

# 整理testing set
df_test_1$Order_Date <- as.numeric(df_test_1$Order_Date)
df_test_1$people_amount <- as.numeric(df_test_1$people_amount)
df_test_1$Begin_Date <- as.numeric(df_test_1$Begin_Date)
df_test_1$days <- as.numeric(df_test_1$days)
df_test_1$price <- as.numeric(df_test_1$price)
df_test_1$PreDays <- as.numeric(df_test_1$PreDays)
df_test_1$preMinutes.去程 <- as.numeric(df_test_1$preMinutes.去程)
df_test_1$preMinutes.回程 <- as.numeric(df_test_1$preMinutes.回程)
df_test_1$Total_Amount_People <- as.numeric(df_test_1$Total_Amount_People)
for (var in one_hot_features){
  df_test_1[,var] <- as.numeric(df_test_1[,var]) 
}
# 計算預測值 (get prediction)
df_test_2 <- df_test_1
df_test_1 <- xgb.DMatrix(data = df_test_1 %>% select(-c(order_id,deal_or_not,group_id,product_name,src_dst.去程,src_dst.回程)) %>% as.matrix()) 
df_test_2$deal_or_not <- predict(bst, df_test_1)
df_test_2 <- df_test_2[,c("order_id", "deal_or_not")]

#改儲存位置 + 改時區 + 存檔 
setwd("/Users/chenchingchun/Desktop/kebuke/answer_set")
time_now <- format(Sys.time(),"%Y%m%d%H%M")
write.csv(df_test,paste0(time_now,"_test.csv"), fileEncoding = "utf-8", row.names = FALSE)