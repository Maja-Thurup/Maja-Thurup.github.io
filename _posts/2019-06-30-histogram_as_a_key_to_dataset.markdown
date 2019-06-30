---
layout: post
title:      "Histogram as a key to Dataset"
date:       2019-06-30 20:42:55 +0000
permalink:  histogram_as_a_key_to_dataset
---


Histogram is a key to initial understanding of the dataset.  It is a simple and powerful visualization tool which is always used for exploratory analysis. 

Basically  histogram is a graph that takes in only one numerical variable and represent its distribution over the dataset. X-axis shows what given variable represents. It is divided into groups or bins. Y-axis shows how frequently certain value occur in each bin. 

There are a lot of ways to create a histogram in Python. For example Pandas, Matplotlib.pyplot and Seaborn are great data visualization libraries. 

Let's import these and plot some histograms. 

We are going to be using movies dataset from imdb.com which contains info from 14k movies. You can download it from here: https://www.kaggle.com/orgesleka/imdbmovies



```
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

Now let's load our data.

```
df = pd.read_csv('imdb.csv')
df.head()
```

Great! Our dataset has some interesting columns. We can build histogram for each column with numeric value. Let's explore the distribution of movie ratings using pandas hist() function.

```
df.imdbRating.hist()
plt.show()
```

![](https://i.imgur.com/fUZa3Ss.png)


Wow. I would think most rating would be around 5.  But most movies are rated between 6 and 8 which is above average. 

Now let's see how we can customize our histogram. We can specify the number of bins to get better view of distribution. 

Matplotlib has a variety of styles. We can  pick the one we like using plt.style.use() function.

Also we can add titles to our x and y axis. In order to do that we can use plt.xlabel() and plt.ylabel() 

Movies have rating from 1 to 10. I want my x axis to show that. We can use plt.xticks() function to specify the number of ticks.

```
plt.style.use('seaborn')
df.imdbRating.hist(bins=20)
plt.xticks(range(0, 11))
plt.title('Movie Ratings distribution')
plt.xlabel("Rating", fontsize=15)
plt.ylabel("Number of movies",fontsize=15)
plt.show()
```

![](https://i.imgur.com/dmz9bpS.png)

Looks much more professional.

Now let's plot a histogram using Seaborn. This is an amazing visualization tool and  is more user friendly than matplotlib. It has a number of built in functions. 

It automatically chooses the right number of bins and also adds a density curve on top of histogram.

```
sns.distplot(df.imdbRating)
plt.xticks(range(0, 11))
plt.show()
```

![](https://i.imgur.com/dJD6sTt.png)

We can also plot two histograms on top of each other for comparison. Let's compare comedy and horror. On our graph we will be able to see which genre has bigger number of movies and higher ratings. 
We need to add plt.legend() function so we could see which genre each histogram represents. 

```
df1 = df[df['Horror'] == True]
df2 = df[df['Comedy'] == True]
sns.distplot(df1.imdbRating, bins=20, kde=False, label='Horror', color='r')
sns.distplot(df2.imdbRating, bins=20, kde=False, label='Comedy')
plt.xticks(range(0, 11))
plt.legend()
```

![](https://i.imgur.com/tF5VZk7.png)

That's interesting. There are more comedies than horror movies. Also comedies are generally higher rated. 

I recommend to use histogram always when you start working with new data because it is a great and simple way to get sense of the data dimension and distribution.





