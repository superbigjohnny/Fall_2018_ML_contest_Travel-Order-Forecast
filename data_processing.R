library(xgboost)
library(stringr)
library(dplyr)
library(ggplot2)
library(Matrix)
library(caret)
library(dummies)
library(InformationValue)
# load dataset
##設定顯示utf-8 Sys.setlocale(category = "LC_ALL", locale = "zh_TW.UTF-8")
df_order = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/order.csv",stringsAsFactors = F)
df_group = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/group.csv",stringsAsFactors = F)
df_airline = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/airline.csv",stringsAsFactors = F)
df_airport_timezone = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/airport_timezone.csv",stringsAsFactors = F)
df_day_schedule = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/day_schedule.csv", header = T,stringsAsFactors = F)
df_train = read.csv("/Users/chenchingchun/Desktop/kebuke/training-set.csv")
df_test = read.csv("/Users/chenchingchun/Desktop/kebuke/testing-set.csv",stringsAsFactors = F)
df_cache = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/cache_map.csv",stringsAsFactors = F)

#data processing
month = data.frame(name_char = c('Jan',"Feb",'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'),
                   name_num  = c('01', "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"))
Convert_Date <- function(x){
  Year= paste0('20',str_sub(x,-2))
  Month=month$name_num[month$name_char == str_sub(x,-6,-4)]
  Day=str_sub(x,1,-8)
  return(as.Date(paste0(Year,'-',Month,'-',Day)))
}




##for group data
for (row in 1:nrow(df_group)) { 
  df_group$Begin_Date[row] <- Convert_Date(df_group$begin_date[row])
}

df_group['SubLine']= as.numeric(substr(df_group$sub_line,15,16))

df_group['Area']= as.numeric(substr(df_group$area,12,13))

df_group['Product_name_nchar'] = nchar(df_group$product_name)

#df_group$good_word = 0  #add good word
#df_group$good_word[grep(paste(words,collapse = "|"),df_group$product_name)] = 1

group_used_cols = c('group_id','Begin_Date','days','Area','SubLine','price',"product_name",'Product_name_nchar'
                    #,"good_word", "promotion_prog"
)

df_group_1 <- df_group[,group_used_cols]
df_order_1 = inner_join(df_order,df_group_1,"group_id")
##for cache data
#df_group$promotion_cache = 0
#for (i in 1:nrow(df_cache)){
#  df_group$promotion_cache[grep(df_cache$url[i],df_group$promotion_prog)] <- i
#  cat("\r",paste0(i/nrow(df_cache)*100,"%"))
#}

df_cache_1 = read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/df_group_cache.csv",stringsAsFactors = F)
df_cache_1$X = NULL
df_order_1 = inner_join(df_order_1,df_cache_1,"group_id")


## for airline data
df_airline[df_airline$group_id %in% c(66808, 1303, 2155, 3103, 5114, 1185, 7766, 1831, 12858, 14022, 67915, 16344, 23282,
                                      25155, 25765, 29299, 33878, 37772, 40535, 41398, 46149, 47180, 48124, 49307, 49801,
                                      50154, 53889, 57939, 58810, 59113, 60100, 61021, 64218, 64395),]

df_airline <- inner_join(df_airline, df_airline %>% 
                           group_by(group_id) %>% 
                           summarise(Fly_count = n()), "group_id")
### 飛行(起始-結束)組合 = 三碼 ＋ 三碼代號
df_airline$src_dst <- paste0(substr(df_airline$src_airport,1,3),"-",substr(df_airline$dst_airport,1,3)) 

length(unique(df_airline$src_dst))#共有686種飛行(起始-結束)組合
### 改成台灣時區
df_airline$src <- substr(df_airline$src_airport,1,3)
df_airline$dst <- substr(df_airline$dst_airport,1,3)
df_airline <- left_join(df_airline, df_airport_timezone[,c("Airport","UTC")],by = c("src" = "Airport"))
df_airline <- left_join(df_airline, df_airport_timezone[,c("Airport","UTC")],by = c("dst" = "Airport"),suffix = c("_src","_dst"))
df_airline$UTC_src <- as.numeric(df_airline$UTC_src) -8
df_airline$UTC_dst <- as.numeric(df_airline$UTC_dst) -8

### 飛行時間(分鐘) =  透過抵達時間(分鐘) - 起始時間(分鐘) + 時差(起始utc - 抵達utc)*60
df_airline$preMinutes <- (as.numeric(as.POSIXlt(df_airline$arrive_time))/60 - as.numeric(as.POSIXlt(df_airline$fly_time))/60) + 60*(df_airline$UTC_src - df_airline$UTC_dst)
### 如果飛行時間為負的，用其他相同起始-抵達地點的平均飛行時間改正
for(i in 1:nrow(df_airline)){
  if (df_airline$preMinutes[i] <= 0){
    df_airline$preMinutes[i] = mean( df_airline$preMinutes[df_airline$src_dst[i] == df_airline$src_dst][df_airline$preMinutes[df_airline$src_dst[i] == df_airline$src_dst] > 0] ) }
}
### 遺失值上網查詢(https://openflights.org/html/apsearch)和錯誤時間typo更改 
df_airline$preMinutes[df_airline$src_dst == "BUD-IST"] = 115
df_airline$preMinutes[df_airline$group_id == "75409f5ea40f84751c73e6465296febb"] = 110  

###把去回程轉成寬資料
df_airline <- reshape(df_airline, idvar = c("group_id","Fly_count"), timevar = "go_back", direction = "wide")

sum(is.na(df_order_1))
# "preMinutes.回程" 239
colnames(df_train_1)[22]
# "src_dst.回程"    239
colnames(df_train_1)[24]

airline_used_cols = c('group_id'
                      ,'preMinutes.去程'
                      ,'preMinutes.回程'
                      ,'src_dst.去程'
                      ,"src_dst.回程"
                      ,"Fly_count"
)

df_airline_1 = df_airline[,airline_used_cols]

df_order_1 = inner_join(df_order_1,df_airline_1,"group_id")


## for order data
#a <- lapply(df_order$order_date,Convert_Date)
#for (row in 1:nrow(df_order)) { 
#  df_order$Order_Date[row] <- a[[row]]
#  cat("\r",paste0(row/nrow(df_order)*100,"%"))
#}
df_order <- read.csv("/Users/chenchingchun/Desktop/kebuke/dataset/df_order_1.csv",stringsAsFactors = F)
df_order <- df_order[,c("order_id", "Order_Date")]
df_order <- df_order[!duplicated(df_order),]

df_order_1 <- inner_join(df_order_1, df_order, "order_id")

df_order_1['Source_1']= as.numeric(substr(df_order_1$source_1,12,13))

df_order_1['Source_2']= as.numeric(substr(df_order_1$source_2,12,13))

df_order_1['Unit']= as.numeric(substr(df_order_1$unit,12,13))

df_order_1['PreDays']= df_order_1['Begin_Date']-df_order_1['Order_Date']

df_order_1['Begin_Date_Month'] = as.numeric(format(as.POSIXlt(as.Date(df_order_1$Begin_Date,origin = "1970-01-01")),"%m"))

df_order_1['Order_Date_Month'] = as.numeric(format(as.POSIXlt(as.Date(df_order_1$Order_Date,origin = "1970-01-01")),"%m"))

df_order_1['Begin_Date_Weekday']= as.POSIXlt(as.Date(df_order_1$Begin_Date,origin = "1970-01-01"))$wday + 1

df_order_1['Order_Date_Weekday']= as.POSIXlt(as.Date(df_order_1$Order_Date,origin = "1970-01-01"))$wday + 1

df_order_1['Return_Date_Weekday']= (as.POSIXlt(as.Date(df_order_1$Begin_Date,origin = "1970-01-01"))$wday + 1 + df_order_1['days'])%%7

df_order_1["Total_price"] = df_order_1["price"] * df_order_1["people_amount"]

df_order_1 <- inner_join(df_order_1,df_order_1 %>%
                           group_by(group_id) %>% 
                           summarise(Total_Amount_People = sum(people_amount),
                                     Amount_Order = n()), "group_id")

df_order_1 <- inner_join(df_order_1, df_order_1 %>%
                           group_by(product_name) %>% 
                           summarise(Amount_Product_Name = n(),
                                     Mean_Predays_by_productname = mean(PreDays),
                                     Median_Predays_by_productname = median(PreDays)), "product_name")

df_order_1 <- inner_join(df_order_1,df_order_1 %>%
                           group_by(Unit) %>%
                           summarise(Count_Unit = n(),
                                     Mean_Amount_Order = mean(Amount_Order),
                                     Median_Amount_Order = median(Amount_Order),
                                     Sum_Amount_Order  = sum(Amount_Order),
                                     Mean_Predays_by_unit = mean(PreDays),
                                     Median_Predays_by_unit = median(PreDays)), "Unit")

df_order_1 <- inner_join(df_order_1,df_order_1 %>%
                           group_by(Area) %>%
                           summarise(Count_area = n(),
                                     Mean_total_price_by_area = mean(Total_price),
                                     Median_total_price_by_area = median(Total_price)), "Area")

df_order_1 <- inner_join(df_order_1,df_order_1 %>%
                           group_by(Source_1) %>%
                           summarise(Mean_total_price_by_source1 = mean(Total_price),
                                     Median_total_price_by_source1 = median(Total_price)), "Source_1")

order_used_columns=c('order_id', 'group_id','Order_Date', 'Source_1', 'Source_2', 'Unit','people_amount',
                     'Begin_Date','days', 'Area', 'SubLine', 'price',"product_name",'PreDays',
                     'Begin_Date_Weekday', 'Order_Date_Weekday', 'Return_Date_Weekday',"Begin_Date_Month", 'Order_Date_Month',
                     'preMinutes.去程','preMinutes.回程','src_dst.去程',"src_dst.回程","Fly_count",
                     "promotion_cache","Product_name_nchar","Total_price",
                     "Total_Amount_People", "Amount_Order",
                     "Amount_Product_Name","Mean_Predays_by_productname","Median_Predays_by_productname",
                     "Count_Unit","Mean_Amount_Order","Median_Amount_Order","Sum_Amount_Order","Mean_Predays_by_unit","Median_Predays_by_unit",
                     "Count_area","Mean_total_price_by_area","Median_total_price_by_area",
                     "Mean_total_price_by_source1","Median_total_price_by_source1"
)

df_order_2 = df_order_1[,order_used_columns]
# One_Hot_encoding
#### unit
df_order_2 <- cbind(df_order_2,dummy(df_order_2$Unit))
Unit_name = NULL
for(i in 1:length(unique(df_order_2$Unit))){
  yo <- paste0("Unit_name_",i)
  Unit_name <- c(Unit_name,yo)
}
colnames(df_order_2)[(dim(df_order_2)[2] - length(unique(df_order_2$Unit))+1):dim(df_order_2)[2]] <- Unit_name
df_order_2$Unit = NULL
#### Amount_Order
df_order_2 <- cbind(df_order_2,dummy(df_order_2$Amount_Order))
Amount_Order = NULL
for(i in 1:length(unique(df_order_2$Amount_Order))){
  yo <- paste0("Amount_Order_",i)
  Amount_Order <- c(Amount_Order,yo)
}
colnames(df_order_2)[(dim(df_order_2)[2] - length(unique(df_order_2$Amount_Order))+1):dim(df_order_2)[2]] <- Amount_Order
#df_order_2$Amount_Order = NULL
#### Total_Amount_People
df_order_2 <- cbind(df_order_2,dummy(df_order_2$Total_Amount_People))
Total_Amount_People = NULL
for(i in 1:length(unique(df_order_2$Total_Amount_People))){
  yo <- paste0("Total_Amount_People_",i)
  Total_Amount_People <- c(Total_Amount_People,yo)
}
colnames(df_order_2)[(dim(df_order_2)[2] - length(unique(df_order_2$Total_Amount_People))+1):dim(df_order_2)[2]] <- Total_Amount_People
#df_order_2$Total_Amount_People = NULL
#### source_1
df_order_2 <- cbind(df_order_2,dummy(df_order_2$Source_1))
Source_1 = NULL
for(i in 1:length(unique(df_order_2$Source_1))){
  yo <- paste0("Source_1_",i)
  Source_1 <- c(Source_1,yo)
}
colnames(df_order_2)[(dim(df_order_2)[2] - length(unique(df_order_2$Source_1))+1):dim(df_order_2)[2]] <- Source_1
df_order_2$Source_1 = NULL
#### source_2
df_order_2 <- cbind(df_order_2,dummy(df_order_2$Source_2))
Source_2 = NULL
for(i in 1:length(unique(df_order_2$Source_2))){
  yo <- paste0("Source_2_",i)
  Source_2 <- c(Source_2,yo)
}
colnames(df_order_2)[(dim(df_order_2)[2] - length(unique(df_order_2$Source_2))+1):dim(df_order_2)[2]] <- Source_2
df_order_2$Source_2 = NULL
#### subline欄位做one-hot encoding
df_order_2 <- cbind(df_order_2,dummy(df_order_2$SubLine))
SubLine <- NULL
for (i in 1:length(unique(df_order_2$SubLine))){
  yo <- paste0("SubLine_",i)
  SubLine <- c(SubLine,yo)
}
colnames(df_order_2)[(dim(df_order_2)[2] - length(unique(df_order_2$SubLine))+1):dim(df_order_2)[2]] <- SubLine
df_order_2$SubLine = NULL

# train/test data
df_train$order_id <- as.character(df_train$order_id)
df_train_1 = inner_join(df_train, df_order_2, 'order_id')

df_test_1  = inner_join(df_test, df_order_2, 'order_id')

# 看類別變數的information value 
iv_value1 = NULL
iv_value2 = NULL
iv_value3 = NULL
for (var in 4:length(colnames(df_train_1))){
  iv_value <- IV(X=as.factor(df_train_1[,colnames(df_train_1)[var]]), Y=df_train_1$deal_or_not)
  iv_value1 <- c(iv_value1,iv_value[1])
  iv_value2 <- c(iv_value2,attr(iv_value,"howgood")[1])
  iv_value3 <- c(iv_value3,length(unique(as.factor(df_train_1[,colnames(df_train_1)[var]]))))
  cat("\r",paste0((var-4)/(length(colnames(df_train_1))-4)*100,"%"))
}
yoyo <- data_frame(colnames = colnames(df_train_1)[4:length(colnames(df_train_1))],
                   Iv_value = round(iv_value1,4),
                   comment  = iv_value2,
                   level_count = iv_value3)