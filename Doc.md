## Notebook Documentation: 9th Place Solution 

### Overview
the solution consists of 3 notebook outputting 3 prediticyions using different models (params)  the process of training, evaluating, and combining predictions from 3 machine learning models: CatBoost and Logistic Regression are used. the solution 
- the second solution for the ensemble was PCA standard scaler and simple imputer, 
- the third ensemble file was using just the catboost model with simple FE,we combined ensembled the second and third solution by mean the result, and the catboost and log-reg was then add to the final mean of the second and third solution. 

### Key Steps
ic.  hh i. 
1. Importing Necessary Libraries
   - Standard Python libraries for data manipulation (pandas, numpy), model training (CatBoostClassifier), and performance evaluation (log_loss from sklearn.metrics).

2. Data Preprocessing
   - Features (X_train_cleaned) and target labels (y_train_cleaned) are loaded or cleaned prior to model training.
   - The test dataset (X_test) is also processed similarly.

3. Training CatBoost Model
   - A CatBoostClassifier model is trained on the preprocessed training data (X_train_cleaned and y_train_cleaned).
   - The model predicts class probabilities on the test set (X_test).
   - Model performance is evaluated using Log Loss.

   
   pred = catBoost_model.predict_proba(X_train_cleaned)
   catBoost_value = log_loss(y_train_cleaned, pred)
   print(f"Log Loss: {catBoost_value}")
   
4. Creating Submission File
   - A submission DataFrame (cat_submission) is created to store predicted probabilities for each class (assuming 125 classes).
   - Predicted probabilities for each class are extracted using the predict_proba method of the CatBoost model.

   
   cat_submission = pd.DataFrame({'id': id})
   columns = [f'Target_{i}' for i in range(125)]
   cat_submission[columns] = 0.001
   
   y_pred_proba_cat = catBoost_model.predict_proba(X_test)
   for i, class_label in enumerate(catBoost_model.classes_):
       cat_submission['Target_' + str(class_label)] = y_pred_proba_cat[:, i]
   
5. Combining Predictions
   - Predictions from the logistic regression model (logistic_sub) are averaged with the predictions from CatBoost (catBoost_sub), creating an ensemble prediction. 
   - The final DataFrame is saved for submission.

   final = (logistic_sub.drop(columns = ['id']) + catBoost_sub.drop(columns = ['id'])) / 2
   final.insert(0, 'id', logistic_sub['id'])
   final.to_csv('mean_logistic_reg_catBoost.csv', index=False)
   
   note that this submission is just one part of three notebooks. the other 2 notebooks used Catboost with different approaches 
### Outputs
- Log Loss: The notebook prints the Log Loss of the CatBoost model, which is a measure of its performance on the training data.
- Submission File: A CSV file (mean_logistic_reg_catBoost.csv) is generated with the final predictions.

---

### Future work 
- Hyperparameter Tuning
- Cross-Validation
- Feature Engineering: Add more feature extraction and selection methods for better predictive accuracy.
