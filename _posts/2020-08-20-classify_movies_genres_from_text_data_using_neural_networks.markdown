---
layout: post
title:      "Classify movies genres from text data using Neural Networks"
date:       2020-08-21 01:38:26 +0000
permalink:  classify_movies_genres_from_text_data_using_neural_networks
---


For this project I [scraped](/Scraper_imdb.ipynb) data from Imdb.com website. Here is link to full [Dataset](https://www.kaggle.com/igorkirko/imdb-summary-trivia).

The goal of this project is to try and see if it is possible to train Neural Network to classify movie genres using movie description as input and list of genres as output. Please refer to my [github page](/Project_genre_final.ipynb) for code.

### Multi label classification problem

Movies often have multiple genres, such as adventure in a spy movie, or crime in a science fiction movie. Usually one genre is predominant. For example drama listed as genre for about 60 % of all movies. And if you look deeper all movies are essentially dramas.

![](https://i.imgur.com/0rEmjFJ.png)

So you can't classify movies by single genre. There are 22 unique genres on imdb and every movie has three or more genres. 

### Quality vs Quantity

The data I am going to use is scraped from IMDB. There are two movie descriptions on the website:

**Short summary** 
- Length	     50-200 symbols
- Quantity     90k items    


**Synopsis** 
- Length	       500-10 000 symbols
- Quantity      11k items	


Dataset contains much more summaries while synopsises contain more information. Lets experiment with both and see which one works better. To solve this problem I am going to use Recurrent Neural Networks (RNN) with Keras. RNNs are good for recognizing patterns in text data.

### Clean and prepare Data

First of all I need to have a clean text. That means to get rid of punctuation and stopwords. Clean text should look like this:

![](https://i.imgur.com/9XUxZSR.png)

Next step is to build vocabulary from text corpus and assign vectors to words usind Word2vec library. Also keras require to have embedding matrix under weights parameter in the embedding layer. Embedding matrix consists of all words in the text corpus with the correspondding vectors. Note that w2v dimension is the size of embedding layer.

Then One hot encode genres and transform texts into sequences of numbers. All input values have to be the same dimesion so we need to pad all sequences to match the longest text.

### Modelling

I am going to use Bidirectional LSTM as hidden layer. Bidirectional LSTMs are an extension to typical LSTMs that can enhance performance of the model on sequence classification problems. Here is the model architecture:

![](https://i.imgur.com/pNdl890.png)

I experimented with the number of epochs and came to a conclusion that it starts overfitting after about 30 epochs for both datasets:

![](https://i.imgur.com/I08tQ8v.png)
![](https://i.imgur.com/4elQ43V.png)

### Results

Model shows nice validation accuracy. But what does it really mean for multiclass problem. At what point we can say that this genre belongs to this movie. Also I would like to see not only numbers but textual result. 
When we make a prediction for single movie description we get a list with 22 probabilities. 

![](https://i.imgur.com/x9URZTd.png)

Now we need to find a level at we can accept that this genre belongs to this movie. One way to do that is to check model's accuracy on different levels and find the best one. In order to do that I am going to use the F-score on different levels. The F-score is a way of combining the precision and recall of the model, and it is defined as the harmonic mean of the model's precision and recall.

First we apply filter to a list of probabilities. It will assign 1 to numbers larger then threshold value and 0 to the rest. Then we compare it to actuall data using F1score: Below you can see the results for both datasets:

**Short summary:**

![](https://i.imgur.com/XMyix5m.png)

**Synopsis:**

![](https://i.imgur.com/Sud8bsA.png)

And know predictions at 0.3 threshold:

**Short summary:**

![](https://i.imgur.com/aiOd1Jx.png)

**Synopsis:**

![](https://i.imgur.com/YutjX0f.png)


### Conclusion

Trivia model turned out to be slightly better. The results for both models are pretty satisfying. The best threshold turned out to be 0.3 which means genre must get 30% probability of occurence to be assigned to a movie. The F1 score of ~0.6 is pretty good considering number of labels: 22 for tirvia and 21 for synopsis. My notebook on [github](/Project_genre_final.ipynb).














