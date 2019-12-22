---
layout: post
title:      "Hypothesis Testing using Welch's t-test in Python"
date:       2019-12-22 18:37:16 +0000
permalink:  hypothesis_testing_using_welchs_t-test_in_python
---

## Goal 

For my second project at Flatiron school I was working with sample database know as Northwind database.  It has been around for a while and was made to help students with learning and understanding what Hypothesis testing is. Although Northwind is a fictional company we a going to treat is as real one. The overall goal of the project was to apply my knowledge of hypothesis testing to Northwind dataset in order to get analytical insights that will help company to increase their revenue.
The main question I had to answer using this database was: Does a discount have statistically significant effect on order quantity?  

## Tools and preparation
All my calculations I did in Python using Jupyter notebook. 
Here are the libraries I used to complete my project:
```
import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
import matplotlib.pyplot as plt 
import seaborn as sns 
```

I transformed all SQL tables into Pandas dataframes. This way it was easier for me to access and use them.
```
conn = sqlite3.connect('Northwind_small.sqlite')
cursor = conn.cursor()

table_names = [res[0] for res in cursor.execute('''select name from sqlite_master where type='table';''').fetchall()]
table_names[5] = '[Order]' # Order is an execution name, so we need to put it into square brakets in order not to get errors in future
```

Create function that makes dataframes:
```
def load_df(table_name=None, conn=None): 
    query = '''select * from {};'''.format(table_name)
    df = pd.read_sql(query, conn)
    return df
```
Store all dfs in a dictionary:
```
data = {}
for table in table_names:
    data[table] = load_df(table, conn)
```

Now I can easily make pandas dataframe for every table I need.

## Hypothesis statement
First as a part of Experiment design we need to state our Null and Alternative hypothesis. Null hypothesis is usually a conservative choice that claims that nothing changed while Alternative hypothesis claims the opposite and it is something we are trying to prove. So our Ho and Ha sound like this:

*  Null Hypothesis (Ho): Discount has no effect on the number of products per order.

*  Alternative Hypothesis (Ha): Discount has an effect on the number of products per order.

## Significance level
Next we need to establish a level of Significance or Alpha level (α). It represents a probability of getting results by pure chance.  I am going to use two tailed test because I only need to know if there is a significant difference between two means. For two tailed test I will set alpha level to 0.025 or 0.25%. That means I am comfortable with 0.25% probability of rejecting Null hypothesis when it is actually true.

## Welch's T-test and size effect
For this project I am going to use Welch's T-test. This test is most suitable for samples with unequal sizes and variances.  To perform this test I am going to use `ttest_ind` function from `scipy` library.
In case that there is a statistical difference between two samples I need to know how big is it. I am going to measure it using Cohen's D formula. It is a value ranged from 0 to 1 where values of 0 - 0.2 have little or no effect and values > 0.8 have a large effect size. Here is this formula in code:

```
def Cohen_d(group1, group2):

    diff = group1.mean() - group2.mean()
		
    n1, n2 = len(group1), len(group2)
		
    var1 = group1.var()
    var2 = group2.var()
		
    # Calculate the pooled threshold
		
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
		
    # Calculate Cohen's d statistic
		
    d = diff / np.sqrt(pooled_var)
		
    return abs(d)
```

## Testing

The next part is setting up control and experimental groups. I am going to split dataset into two categories:

* Control: customers who did not receive a discount
* Experimental: customers who received a discount

```
no_disc = orderdetail_df[orderdetail_df['Discount'] == 0]['Quantity']
disc = orderdetail_df[orderdetail_df['Discount'] != 0]['Quantity'] 
```
I set equal variance parameter to False in order to perform Welch's T-test.
```
t_stat, p = stats.ttest_ind(no_disc, disc, equal_var=False)
d = Cohen_d(no_disc, disc)

 print('p-value', p)
 print('Reject Null Hypothesis') if p < 0.025 else print('Failed to reject Null Hypothesis')
 print("Cohen's d:", d)
```

Here are the results:
```
p-value 5.65641429030433e-10
Reject Null Hypothesis
Cohen's d: 0.2862724481729283
```

## Conclusion

P-value is very small and Cohen's D shows medium effect size so we safely Reject Null Hypothesis and accept the Alternative Hypothesis.  Python and scipy library make it much easier to perform this powerful test. With only few lines of code we are able to gain useful analytical insights, draw conclusions and plan ahead business strategy.

