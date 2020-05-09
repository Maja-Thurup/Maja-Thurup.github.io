---
layout: post
title:      "Movie box office outcome with SVM classifier."
date:       2020-05-09 17:51:26 +0000
permalink:  movie_box_office_outcome_with_svm_classifier
---


For this project, I had to choose a problem that can be solved by classification, find an appropriate dataset, and build a classification model that solves my problem. The problem I chose is:

**Is it possible to predict movie success or failure based on budget, genre, cast and crew information?**

Obviously, there must be something that makes people prefer one movie over another.  Let's look at the example. Aquaman is a blockbuster action-adventure movie with a pg-13 rating, It has popular main actors and overall great production design. With a 150 million dollar budget, It makes over a billion in the box office. Another great recent movie Alita: Battle Angel fails miserably. But why? It has great cast and crew, great production, and the same movie rating. Unsuccessful doesn't necessarily mean bad.  Maybe there is a pattern here that makes people go and watch Aquaman and ignore Alita.
Let's try and find out.

## Preparation.
The dataset I used can be found here [boxofficemojo dataset](https://www.kaggle.com/igorkirko/wwwboxofficemojocom-movies-with-budget-listed).  First I need to deal with null values. The majority of movies have a budget between 14 and 60 million. Since I have huge outliers I will fill null values with median value.

```
df = pd.read_csv('Mojo_data_update.csv')

df.drop(cols_to_drop, axis=1, inplace=True)
df.dropna(subset=['release_date', 'main_actor_2', 'composer', 'producer', 'distributor', 'run_time'], inplace=True)
   
genres = ['genre_1', 'genre_2']
df.dropna(subset=genres, inplace=True)
fill_cin = df[df['cinematographer'].isnull()].director
df.cinematographer.fillna(fill_cin, inplace=True)
 
df.mpaa.fillna('PG-13', inplace=True)
df.budget.fillna(30000000, inplace=True)
```

In order to find if a movie is successful, I need to know the budget and box office data. It is not obvious but all posted box office data is not actual profit. It is how many theatres made. Knowing that theatres generally keep 40-50 % of the revenue the general rule of thumb is: if the box office is two times larger than budget - the movie is successful.

```
df['successful'] = (df.worldwide > df.budget*2)*1
df.drop('worldwide', axis=1, inplace=True)
```
Our dataset has a lot of unique values. As they are mostly categorical we are going to use one-hot encoding to transform our dataset for modeling.  This will result in thousands of new columns. In order to save computational time let's filter our dataset a little bit.

```
def filter_by_freq(df, column, min_freq):
    """Filters the DataFrame based on the value frequency in the specified column.
    input: DataFrame, Column name, Minimal value frequency for the row to be accepted.
    return: Frequency filtered DataFrame.
    """
    # Frequencies of each value in the column.
    freq = df[column].value_counts()
    # Select frequent values. Value is in the index.
    frequent_values = freq[freq >= min_freq].index
    # Return only rows with value frequency above threshold.
    return df[df[column].isin(frequent_values)]
```

This function will make our dataset more suitable for modeling. I feel like the main actor and director are the two most important figures in production. Let's filter our dataset by main actor and director that appear at least twice.

```
cols = ['director', 'main_actor_1']
for col in cols:
    
    df_world = filter_by_freq(df_world, col, 2)
```

## Modelling
First, let's make our test and train sets.
```
from sklearn.model_selection import train_test_split

X = df_world.drop('successful', axis=1)
y = df_world['successful']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=111)
```
Most of our columns are categorical so we need to use OneHotEncoder. Also we need to standartize budget and run_time columns.
```
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

preprocessor = make_column_transformer((StandardScaler(), ['budget', 'run_time']), 
                                           remainder=OneHotEncoder(handle_unknown='ignore', sparse=False))
```
Let's set up our pipeline. Besides preprocessor, I am going to include PCA into my pipeline. Principal Component Analysis is basically a statistical procedure to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables. I will set the n_components parameter to .9 which means it will keep the number of components that explain 90% of the variance. Explained variance is the part of the model's total variance that is explained by factors that are actually present and isn't due to error variance. Overall it is all needed to save computational time by reducing dataset dimensionality and saving the most valuable information at the same time.

```
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import svm

svm_pipe = Pipeline([('preprocessor', preprocessor),
                         ('pca', PCA(n_components=0.9)),
                         ('clf', svm.SVC())])
```

I tested this dataset with different models and SVM showed the best results with default parameters. The linear SVM classifier works by drawing a straight line between two classes. All the data points that fall on one side of the line will be labeled as one class and all the points that fall on the other side will be labeled as the second. So it works well on binary data. Now let's try and tune our model using gridsearch.

```
from sklearn.model_selection import GridSearchCV

params = [{'clf__C': [0.1, 1, 10, 100], 'clf__gamma': [10, 1, 0.1, 0.01], 'clf__kernel': ['sigmoid']},
                            {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': [10, 1, 0.1, 0.01], 'clf__kernel': ['rbf']},
                            {'clf__C': [0.1, 1, 10, 100,], 'clf__gamma': [10, 1, 0.1, 0.01], 'clf__degree': [3,4,5], 'clf__kernel': ['poly']}]

grid_search = GridSearchCV(svm_pipe, params)
grid_search.fit(X, y)
print(grid_search.best_params_)
```

```
{'clf__C': 100, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
```

Finally let's try our model with new parameters. I am going to use accuracy_score and classification report to evaluate model performance.

```
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

svm_best = Pipeline([('preprocessor', preprocessor),
                         ('pca', PCA(n_components=0.9)),
                         ('clf', svm.SVC(C=100, gamma=0.1, kernel='rbf'))])

svm_best.fit(X_train, y_train)
training_preds = svm_best.predict(X_train)
test_preds = svm_best.predict(X_test)

# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)
class_report = classification_report(y_test, test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))
print('---------------------------------------------------------')
print(class_report)
```
```
Training Accuracy: 99.97%
Validation accuracy: 79.63%
---------------------------------------------------------
                             
              precision    recall  f1-score   support

           0       0.81      0.92      0.86      1093
           1       0.75      0.52      0.61       488

    accuracy                           0.80      1581
   macro avg       0.78      0.72      0.74      1581
weighted avg       0.79      0.80      0.78      1581
``` 

Great! Almost 80% accuracy is not bad. Turns out there is a pattern. Further work would be testing our model with different cast and crew members looking for features that affect movie box office the most.
