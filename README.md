# Feature-Importance-Identifiers
In this repository, I share the code I identified to quickly determine features and their importance, specifically in random forest models. I did not find anything identical or similar to this via search engines (Google).

# The Story
While working on one of my projects in my Data Science Bootcamp, I encountered a complication when looking at feature importance in machine learning models. 

I found code to use which would easily match up features with their importance scores but only given column key numbers for the features, rather than the name of the columns; I had difficulty efficiently identifying features in my models, especially in large datasets. I was left having to manually count my feature's columns to determine, an exhausting process.

Reaching out to my cohort, TAs, and of course, Google, I was unable to find an easier way to achieve my goal. However, upon discussing with a fellow student, I created code that would fix my issue.

I create this demo using data from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction), which contains house sales prices for King County in Washington, USA.

# First

The feature_importances_ works with Random Forest models in both regression and classifiers. In this demo, I will use it for regression. Before I could look at importance scores, I had to follow the standard steps needed for creating a random forest model: run imports; upload my dataset with pandas; clean data (my data was already clean); assign my target (y) and features (X); perform a train-test split; import the random forest model; and fit the model on the training data. Running R^2 scores is not required to view importance scores, but I ran it anyway. You can see all of the steps performed in my Colaboratory/raw code.

# BEFORE:

```rf.feature_importances_```

the above creates output as an array: 

```
array([0.00286812, 0.01046085, 0.25133288, 0.01360806, 0.00193086,
       0.02964955, 0.01061545, 0.00296464, 0.33616096, 0.02075849,
       0.00532124, 0.03731132, 0.00222625, 0.01509868, 0.15103039,
       0.06257352, 0.03294906, 0.01313967])
```

The scores are in the array in order of the columns within our features in their basic form. “Feature_importances_” on its own is not necessarily a user-friendly way to correlate them.

Thanks to our pal, Google, I stumbled upon [this article](https://machinelearningmastery.com/calculate-feature-importance-with-python/) which gave me a way to at least match up the features to their importance scores. In the article, however, they used coefficients to calculate feature importance rather than feature_importances_.

```
importance = rf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```
 

Output:

```
Feature: 0, Score: 0.00287
Feature: 1, Score: 0.01046
Feature: 2, Score: 0.25133
Feature: 3, Score: 0.01361
Feature: 4, Score: 0.00193
Feature: 5, Score: 0.02965
Feature: 6, Score: 0.01062
Feature: 7, Score: 0.00296
Feature: 8, Score: 0.33616
Feature: 9, Score: 0.02076
Feature: 10, Score: 0.00532
Feature: 11, Score: 0.03731
Feature: 12, Score: 0.00223
Feature: 13, Score: 0.01510
Feature: 14, Score: 0.15103
Feature: 15, Score: 0.06257
Feature: 16, Score: 0.03295
Feature: 17, Score: 0.01314
```


With the above, the article also suggested a way to graph our findings to display our features visually.

```
plt.bar([x for x in range(len(importance))], importance);
plt.xlim(-0.5,17.5);
```
Output:

![features graph 1](https://user-images.githubusercontent.com/86759538/133001723-7b31265a-cd60-417d-84dd-20dec3ab8720.png)

Looking online, the above was the easiest way I could find that would match up features with their scores. It is a great deal easier to analyze than an array, but I still strived to achieve a higher level of user-friendliness.

To identify which feature was each number, I manually counted out each column key in our features.

```X.head(1)```

Output:
![Screenshot 2021-09-12 at 4 50 15 PM](https://user-images.githubusercontent.com/86759538/133002398-10060252-2644-419b-8544-d22bf553fa17.png)

From this, I needed to count manually 0 to 17 by column to identify the highest importances: 8 is grade; 2 is sqftliving; 14 is latitude. This dataset's features weren't few but weren't large either. This process would not be efficient in a dataset with more columns.

# After
While  discussing with a [classmate](https://github.com/Drodricks0), I starting playing around with code and was able to find a way to solve my problem by creating a mini-data frame. By doing so, I would achieve what I was looking to do: identify the feature importances readily.

The code:
```
data = {'Feature': X.columns, 'Importance': rf.feature_importances_}
feature_importances = pd.DataFrame(data)
feature_importances
```
Output:

![Screenshot 2021-09-12 at 4 54 57 PM](https://user-images.githubusercontent.com/86759538/133002515-bb14cca6-3424-4b87-b2ca-683b33b559f0.png)

I was happy with just this table, but by creating a visualization as we did before, it becomes immediately apparent which features are the most important. Upon sharing my above code, one of my Bootcamp TAs shared code with me to visualize my findings without the table. As I like having the numerical chart, I adapted her code to fit with my code.

```
rf_features = pd.DataFrame({'feature': X_train.columns, 'feature_importance': rf.feature_importances_})
rf_features.sort_values(by = 'feature_importance', inplace = True)

plt.figure(figsize = (12, 5))
plt.barh(rf_features['feature'], rf_features['feature_importance'])
plt.xlabel('Importance Score', fontsize = 13)
plt.xticks(fontsize = 12)
plt.ylabel('Features', fontsize = 13)
plt.yticks(fontsize = 12)
plt.title('Feature Importances in our Model', fontsize = 14);
```

![feature importances graph 2](https://user-images.githubusercontent.com/86759538/133002585-729baed7-19b9-4893-95a3-41995af2e0fd.png)

My "After" code in the last section is a great way to make it easier to visualize and effortlessly identify the most important features of our models.

My technique works for any model which allows feature_importances_, i.e., Random Forests. I hope this code can assist others with feature_importances_ in addition to discovering more applications.

