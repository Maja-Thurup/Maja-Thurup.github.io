---
layout: post
title:      "My first Kaggle competition"
date:       2020-09-20 21:12:26 +0000
permalink:  my_first_kaggle_competition
---


So after I finished my capstone project at Flatiron school I decided to test my Data science skills and take part in one of the Kaggle competitions. The title of the competitions is [Mechanisms of Action (MoA) Prediction](https://www.kaggle.com/c/lish-moa/overview) and the goal is to build and improve the algorithm that classifies drugs based on their biological activity. My goal was simple: create a working notebook so I could appear on the leaderboard. 
The task is a multilabel classification problem. I used all my knowledge to deal with this problem and address all the issues:

- One hot encode categorical features
- Standardized the data
- Used PCA technique
- Built a Neural Network
- Performed a cross-validation

I was getting pretty good results and was proud of myself. The competition didn't seem to be that difficult. To see my score on the leaderboard I needed to submit a working notebook and a test csv file with predictions. It worked but my result was not as good as I hoped. I was placed 700 on the leaderboard which was not the worst result at least. Only then I noticed that difference in accuracy between my score and leader's score was roughly 0.001! I realized that making a good working model is just the beginning. 

That's when the actual competition starts! So many things I need to adjust to try and make my model perfect. Try different components for PCA, for neural network architecture, I need to choose the right number of hidden layers, learning rate, dropout rate, batch size, and a number of neurons. The other big problem is applying regularization when building a neural network: should I apply Dropout layer before or ofter Batch normalization layer. This [paper](https://arxiv.org/pdf/1801.05134.pdf) suggests that we need to apply Dropout after all BN layers but a fare number of other sources say otherwise. So it all comes down to finding what works best for this particular model. 

After a couple of days of tuning and tweaking my model I was able to improve my result by 0.0005 which now feels like a huge success for me. Still, there are a lot of things that I need to do and learn. The competition allows only three submissions per day so you need to choose carefully what you want to submit. The test set on which model is evaluated is only 25% of the actual test data. The other 75%  will be used for final scores.

What I learned during this competition:

- Visualizations are very important and give a good perspective
- Always perform cross-validation
- Hyperparameter tuning function is a must
- The most precise model on the validation set doesn't mean the best on the final test set.
- Read Discussions and public notebooks

Overall taking part in a competition is a good place to practice and learn new tricks and techniques. Every dataset requires time and effort to get a good understanding of what it is about. Sometimes it almost feels like art and it is very satisfying when you can make your model work and produce a good result.




