## Ivan Valiente
  
 +### April 1st 2017"
 +### April 1st 2017

## Executive Summary
--------------------

The goal of this project is to build a model that predicts the type of
"personal activity" using measurements from accelerometers on the belt,
forearm, arm, and dumbell of 6 individuals.

The participants in the data colection were asked to perform barbell
lifts correctly and incorrectly in 5 different ways.

The candidate-models will be trained to predict the "classe" variable in
the training set, by using any other variables in the data set.

The final prediction model will be used to predict 20 different test
cases provided in the "test" data set.

### The Data Set

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

More information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

### Report summary

This report is split in the three following sections:

1.  Getting and cleaning data
2.  Models benchmarking and final model selection
3.  Prediction of the 20 "classe" variables by running the final model
    on the test data set

2) Getting, Cleaning, and Preprocessing Data
--------------------------------------------

### Downloading the data sets

    if(!file.exists("./project")){
        dir.create("./project")
        fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

        download.file(fileUrl1, 
                     destfile = "./project/pml-training.csv")
        
        fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

        download.file(fileUrl2, 
                     destfile = "./project/pml-testing.csv")
     
    }
    list.files("./project")

    ## [1] "pml-testing.csv"  "pml-training.csv"

    training <- read.csv("./project/pml-training.csv")
    testing <- read.csv("./project/pml-testing.csv")

### Data cleaning

The first cleaning step is to remove the colons with non-available (NA)
data. The goal of this step is to reduce the risk of getting unexpected
behaviors or errors of the models due to missing data.

    ##str(training)
    testing <- testing[,colSums(is.na(testing))==0]
    ##names(testing[,colSums(is.na(testing))==0])
    ##
    names_training <- names(testing)
    names_training[length(names_training)] <- "classe"
    training <- training[,names_training]

    ## Checking that there is no colon with missing data in the training set
    names(training[,colSums(is.na(training))>0])

    ## character(0)

    names(training)

    ##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
    ##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
    ##  [7] "num_window"           "roll_belt"            "pitch_belt"          
    ## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
    ## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
    ## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
    ## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
    ## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
    ## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
    ## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
    ## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
    ## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
    ## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
    ## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
    ## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
    ## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
    ## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
    ## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
    ## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
    ## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"

As the goal of the model is to predict the type of activity, one should
select, among the variables provided in the data set, those that are
closely related to motion activity. That means to include in the model
only the variables linked to the acceleration measurements, and to
exclude all those variables that are not physically related to motion
activity.

From the variable names I can infer that the first 7 variables in the
data sets ("X", "user\_name", "raw\_timestamp\_part\_1",
"raw\_timestamp\_part\_2", "cvtd\_timestamp", "new\_window",
"num\_window") are not related to any type of motion the model should
predict. So they can be excluded from this model study.

    library(dplyr)

    ## Warning: package 'dplyr' was built under R version 3.2.5

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

    training <- select(training,-new_window,-cvtd_timestamp,
                       -raw_timestamp_part_1,-raw_timestamp_part_2,-user_name,-num_window,-X)
    testing <- select(testing, -new_window,-cvtd_timestamp,
                       -raw_timestamp_part_1,-raw_timestamp_part_2,-user_name,-num_window,-X)

### Data Splitting

In the following step the initial training set is split in two subsets.

The "traiining\_subset" will be used for model training. The
"validation\_subset" provides the mean to evaluate and compare the
different models accuracy.

    library(caret)

    ## Warning: package 'caret' was built under R version 3.2.5

    ## Loading required package: lattice

    ## Warning: package 'lattice' was built under R version 3.2.5

    ## Loading required package: ggplot2

    set.seed(7826)
    inTrain = createDataPartition(training$classe, p = 0.7, list = FALSE)

    training_subset = training[ inTrain,]
    validation_subset = training[-inTrain,]

    ## Verification of the randomisation of the two data sub-sets

    dim(training_subset)

    ## [1] 13737    53

    dim(validation_subset)

    ## [1] 5885   53

    table(training_subset$classe)/dim(training_subset)[1]

    ## 
    ##         A         B         C         D         E 
    ## 0.2843416 0.1934920 0.1744195 0.1639368 0.1838101

    table(validation_subset$classe)/dim(validation_subset)[1]

    ## 
    ##         A         B         C         D         E 
    ## 0.2844520 0.1935429 0.1743415 0.1638063 0.1838573

### Effects of Preprocessing with Principal Components Analysis

During this project assignment I will also investigate the possibility
to reduce the number of variables by removing the variables that are
highly correlated. I will test PCA preprocessing, by evaluating the
impact it has in the models' accuracy.

For this invastigation I fixed the "thresh" variable at 99% in order to
keep in the Principal Component variables the maximun of variance of the
initial training variables.

    ##
    numeric_subset <- select(training_subset,-classe)

    prComp <- preProcess(numeric_subset, method="pca", thresh = .99)
    prComp

    ## Created from 13737 samples and 52 variables
    ## 
    ## Pre-processing:
    ##   - centered (52)
    ##   - ignored (0)
    ##   - principal component signal extraction (52)
    ##   - scaled (52)
    ## 
    ## PCA needed 37 components to capture 99 percent of the variance

    trainPC <- predict(prComp,training_subset)

    validationPC <- predict(prComp,validation_subset)
    predictionPC <- predict(prComp,testing)

3) Moldel Benchmarking & Model Selection
----------------------------------------

In this section I will assess and compare the accuracy of four different
models. The accuracy of the models is estimated on validation subset.

The models that will be tested are the following:

1.  Random forest decision trees (rf)
2.  Stochastic gradient boostting trees (gbm)
3.  Support vector machine (svm)
4.  Decision trees with CART (rpart)

For all the tree-based classification models I will use k-fold
cross-validation setting k=5.

    fitControl <- trainControl(method = "cv", number = 5)





    modFit_rf <- train(classe ~ . , data = training_subset, method="rf", trControl = fitControl)

    ## Loading required package: randomForest

    ## Warning: package 'randomForest' was built under R version 3.2.5

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    modFit_rpart <- train(classe ~ . , data = training_subset, method="rpart", trControl = fitControl)

    ## Loading required package: rpart

    library(gbm)

    ## Warning: package 'gbm' was built under R version 3.2.5

    ## Loading required package: survival

    ## Warning: package 'survival' was built under R version 3.2.5

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.1

    modFit_gbm <- train(classe ~ . , data = training_subset, method="gbm", trControl = fitControl,
                        verbose = FALSE)

    ## Loading required package: plyr

    ## -------------------------------------------------------------------------

    ## You have loaded plyr after dplyr - this is likely to cause problems.
    ## If you need functions from both plyr and dplyr, please load plyr first, then dplyr:
    ## library(plyr); library(dplyr)

    ## -------------------------------------------------------------------------

    ## 
    ## Attaching package: 'plyr'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     arrange, count, desc, failwith, id, mutate, rename, summarise,
    ##     summarize

    library(e1071)

    ## Warning: package 'e1071' was built under R version 3.2.5

    modFit_svm <- svm(classe ~ ., data = training_subset) 


    validation_rf <- predict(modFit_rf,validation_subset)
    validation_gbm <- predict(modFit_gbm,validation_subset)
    validation_svm <- predict(modFit_svm,validation_subset)
    validation_rpart <- predict(modFit_rpart,validation_subset)


    cm_svm <- confusionMatrix(validation_svm,validation_subset$classe)
    cm_rf <- confusionMatrix(validation_rf,validation_subset$classe)
    cm_gbm <- confusionMatrix(validation_gbm,validation_subset$classe)
    cm_rpart <- confusionMatrix(validation_rpart,validation_subset$classe)


    ## model fit with principal components predictors

    modFit_rf_PC <- train(classe ~ . , data=trainPC, method="rf", trControl = fitControl)
    modFit_gbm_PC <- train(classe ~ . , data=trainPC, method="gbm", trControl = fitControl,
                        verbose = FALSE)
    modFit_svm_PC <- svm(classe ~ ., data = trainPC) 

    validation_gbm_PC <- predict(modFit_gbm_PC,validationPC)
    validation_svm_PC <- predict(modFit_svm_PC,validationPC)
    validation_rf_PC <- predict(modFit_rf_PC,validationPC)


    cm_svm_PC <- confusionMatrix(validation_svm_PC,validation_subset$classe)
    cm_rf_PC <- confusionMatrix(validation_rf_PC,validation_subset$classe)
    cm_gbm_PC <- confusionMatrix(validation_gbm_PC,validation_subset$classe)

    AccuracySummary <- data.frame( Model = c("rf","gbm","svm","rpart","rf_PC","gbm_PC","svm_PC"),
                                   Accuracy = rbind(cm_rf$overall[1],cm_gbm$overall[1],cm_svm$overall[1],
                                                    cm_rpart$overall[1],cm_rf_PC$overall[1],cm_gbm_PC$overall[1],
                                                    cm_svm_PC$overall[1]))
    print(AccuracySummary)

    ##    Model  Accuracy
    ## 1     rf 0.9915038
    ## 2    gbm 0.9593883
    ## 3    svm 0.9520816
    ## 4  rpart 0.5004248
    ## 5  rf_PC 0.9792693
    ## 6 gbm_PC 0.8458794
    ## 7 svm_PC 0.9617672

From previous results one can observe:

1.  Random forest is the model that provides the highest accuracy among
    the four models.
2.  Because the accuracy of "rf" is as high as 99.1%, I consider that
    there is no need of model stacking to improve the accuracy.
3.  One also observe that the utilisation of principal components (PC)
    reduces the accuracy for all the models considered

3) Final Model Prediction
-------------------------

Based on the model assessment of the previous section I will run the
"rf" model on the testing set to get the predictions for the final quiz
of this MOOC

    prediction_results_rf <- predict(modFit_rf,testing)
    quiz_results_rf <- data.frame(problem_id = testing$problem_id, predicted_classe = prediction_results_rf)
    print(quiz_results_rf)

    ##    problem_id predicted_classe
    ## 1           1                B
    ## 2           2                A
    ## 3           3                B
    ## 4           4                A
    ## 5           5                A
    ## 6           6                E
    ## 7           7                D
    ## 8           8                B
    ## 9           9                A
    ## 10         10                A
    ## 11         11                B
    ## 12         12                C
    ## 13         13                B
    ## 14         14                A
    ## 15         15                E
    ## 16         16                E
    ## 17         17                A
    ## 18         18                B
    ## 19         19                B
    ## 20         20                B
