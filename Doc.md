## Notebook Documentation: 9th Place Solution 

### Overview
the solution consists of 3 notebook outputting 3 prediticyions using different models (params)  the process of training, evaluating, and combining predictions from 3 machine learning models: CatBoost and Logistic Regression are used. the solution 
## `part1.ipynb` this notebook consist of the following 
- reading the Data
- apply isolation forest to mark anamolies 
- modeled using catboost class weight parameter is set to balance the minority classes   
- 
## `part2.ipynb` solution consist of  
- simple imputer   
- applying PCA
- standard scaler (No need to) 
- OneHot Encoding
- class weight is set using catboost

## `part3.ipynb` consist of 0
-fill in missign values with bfill method 
- engineered features and dropping features 
- applied isolation forest 
- modeled using catboost and logistic regression the catboost model include class weight params setting 
 

### metric
The metric used is log loss 
  
4. Creating Submission File
   - A submission DataFrame is created to store predicted probabilities for each class 
   - Predicted probabilities for each class are extracted using the predict_proba method of the CatBoost model.

   
   cat_submission = pd.DataFrame({'id': id})
   columns = [f'Target_{i}' for i in range(125)]
   cat_submission[columns] = 0.001
   
   y_pred_proba_cat = catBoost_model.predict_proba(X_test)
   for i, class_label in enumerate(catBoost_model.classes_):
       cat_submission['Target_' + str(class_label)] = y_pred_proba_cat[:, i]
   
5. Combining Predictions
   - Predictions from `part1.ipynb`, `part2.ipynb`
are averaged and then again averaged with the predictions from `part3.ipynb`, creating an ensemble prediction. 
  
  
### Outputs

- Submission File: A CSV file (mean_of_all.csv) is generated with the final predictions.

---

### Future work 
- Hyperparameter Tuning
- Cross-Validation
- Feature Engineering: Add more feature extraction and selection methods for better predictive accuracy.