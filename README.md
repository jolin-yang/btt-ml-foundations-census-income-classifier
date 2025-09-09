# Census Income Classifier

## 📑 Overview:
This was the final project I submitted for Break Through Tech's Machine Learning Foundations course, where I aimed to apply all the machine learning skills/techniques I learned from the course into building an end-to-end ML classification pipeline from a real-life, relevant dataset. In this project, my goal was to use the Adult Census Income dataset to develop a binary classifier that predicted whether a person's income exceeded 50K, based on a variety of features like age, years of education, occupation, marital status, and capital gain/loss. To build the classifier, I performed EDA, data preprocessing to handle 30,000+ entries in the data set, feature engineering, model training, validation, and selection - making use of Python ML libraries like Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib to train models and perform data visualizations. 

## 🌐 Relevance 
This supervised ML problem holds real-world relevance because the insights extracted from my findings could help companies better understand consumer behavior/patterns, enabling them to tailor advertising strategies that maximize impact and effectiveness on target audiences. Having information about who is likely to make over 50K and who isn't would allow a company to identify future customers who are more likely to afford and purchase its products, helping them focus their advertising efforts on high-potential consumers. For instance, a company that sells luxury, high-end designer products could use this information to concentrate the majority of their advertising on audiences with higher income levels who are more likely to purchase such products.

## 🔧 Toolkit
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn   

## ✅ Model Workflow
- Exploratory Data Analysis (EDA)
  - Inspect the dataset and locate outliers and missing values.  
- Data Preprocessing
  - Clean dataset - replace missing values, winsorize outliers, scale numerical data, and rename the label to more accurately match the problem.
- Feature Engineering
  - One-hot encode categorical values and remove irrelevant features. 
- Model Training
  - For the 4 different types of classifiers, train two of each type:
    - One classifier trained with default hyperparameters
    - One classifier trained with best hyperparameter values found using GridSearchCV
- Model Validation
  - Performed k-fold cross validation using GridSearchCV to identify hyperparameter values optimizing model performance, and tune hyperparameters to those values.    
- Model Testing
  - Test model on the testing data. 
- Model Evaluation
  - Evaluated based on AUC, log loss, and accuracy score.
- Model Selection
  - Select the model that performs the best in regards to evaluation metrics.     

## 📊 Analysis
For all four types of the classification models I trained and tested (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosted Decision Tree), my findings indicated that the classifiers using the best hyperparameter value(s) performed slightly better than their default classifier (set to default hyperparameter values), using AUC, log loss, and accuracy score as my evaluation metrics - though the differences in these metrics between the two classifiers were often miniscule. 

After training and testing all the models, I found a pattern regarding the features of the model: the top 5 features that held the most weight on each model always seemed to be 'age', 'education-num', 'capital-gain', 'capital-loss', and 'hours-per-week' - which in context, does make sense since all of these features are relevant pieces of information that often determine how much someone gets paid. 

Evaluation Metrics for the 8 Models Trained and Tested: 
| Model                                                             |         AUC          |       Log Loss       |      Accuracy      |
|-------------------------------------------------------------------|----------------------|----------------------|--------------------|
| Default Logistic Regression Classifier                            |  0.8982140438549376  |  0.3259699892214959  | 0.8456499488229273 |
| Optimized Logistic Regression Classifier (Best Params)            |  0.898289353851902   |  0.3256612002905544  | 0.8460593654042988 |
| Default Decision Tree Classifier                                  |  0.885664712887934   |  0.3372596798806298  | 0.8458546571136131 |
| Optimized Decision Tree Classifier (Best Params)                  |  0.8900379468976952  |  0.6025988469956406  | 0.8505629477993859 |
| Default Random Forest Classifier                                  |  0.8884487308222218  |  0.5599368820049132  | 0.8460593654042988 |
| Optimized Random Forest Classifier (Best Params)                  |  0.8851148915302525  |  0.333518624318522   | 0.8489252814738997 |
| Default Gradient Boosted Decision Tree Classifier                 |  0.9161695817668076  |  0.29753747587200474 | 0.8616171954964176 |
| **Optimized Gradient Boosted Decision Tree Classifier (Best Params)** |  **0.9234405576442566**  |  **0.28315057269897215** | **0.8665301944728762** |


## 🎯 Main Takeaways
As shown in the table above, evaluating all the classifiers with the specified metrics (AUC, log loss, accuracy) and performing model selection yielded the optimized Gradient Boosted Decision Tree classifier (when all features in the census dataset were used, n_estimators = 100, and max_depth = 6) as the best-performing model across the board. The optimized GBDT model achieved an AUC of 0.923, log loss of 0.283, and 87% accuracy, demonstrating the overall strength of this model.
