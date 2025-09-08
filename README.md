# Census Income Classifier

## 📑 Overview:
This was the final project I submitted for Break Through Tech's Machine Learning Foundations course, where I aimed to apply all the machine learning skills/techniques I learned from the course into building an end-to-end ML classification pipeline from a real-life, relevant dataset. In this project, my goal was to use the Adult Census Income dataset to develop a binary classifier that predicted whether a person's income exceeded 50K, based on a variety of features like age, years of education, occupation, marital status, and capital gain/loss. To build the classifier, I performed EDA, data preprocessing to handle 30,000+ entries in the data set, feature engineering, model training, validation, and selection - making use of Python ML libraries like Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib to train models and perform data visualizations. 

## 🌐 Relevance 
This supervised ML problem holds real-world relevance because the insights extracted from my findings could help companies better understand consumer behavior/patterns, enabling them to tailor advertising strategies that maximize impact and effectiveness on target audiences. Having information about who is likely to make over 50K and who isn't would allow a company to identify future customers who are more likely to afford and purchase its products, helping them focus their advertising efforts on high-potential consumers. For instance, a company that sells luxury, high-end designer products could use this information to concentrate the majority of their advertising on audiences with higher income levels who are more likely to purchase such products.

## 🧮 Classifier Workflow
- Exploratory Data Analysis (EDA)
  - Inspect the dataset and locate outliers and missing values.
- Data Preprocessing
  - Replace missing values, winsorize outliers, scale numerical data, and rename the label to more accurately match the problem.
- Feature Engineering
  - One-hot encode categorical values and remove irrelvant features.
- Model Training
  - For the 4 different types of classifiers, train two of each type:
    - One classifier trained with default hyperparameters
    - One classifier trained with best hyperparameter values found using GridSearchCV
- Model Validation
  - Use GridSearchCV to perform k-fold cross validation to identify hyperparameter values optimizing model performance, and tune hyperparameters to those values.
- Model Testing
  - Test model on the testing data.
- Model Evaluation
  - Evaluated based on AUC, log loss, and accuracy score.
- Model Selection
  - Select the model that performs the best in regards to evaluation metrics.

## 📊 Results 
For all four types of the classification models I trained and tested (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosted Decision Tree), my findings indicated that the classifiers using the best hyperparameter value(s) performed slightly better than their default classifier (set to default hyperparameter values), using AUC, log loss, and accuracy score as my evaluation metrics - though the differences in these metrics between the two classifiers were often miniscule. 

After training and testing all the models, I found a pattern regarding the features of the model: the top 5 features that held the most weight on each model always seemed to be 'age', 'education-num', 'capital-gain', 'capital-loss', and 'hours-per-week' - which in context, does make sense since all of these features are relevant pieces of information that often determine how much someone gets paid. 

Evaluating all the classifiers with the specified metrics (AUC, log loss, accuracy) and performing model selection yielded the optimizied Gradient Boosted Decision Tree classifier (when all features in the census dataset were used, n_estimators = 100, and max_depth = 6) as the best-performing model across the board. The optimized GBDT model achieved an AUC of 0.923, log loss of 0.283, and 87% accuracy, demonstrating the overall strength of this model.
