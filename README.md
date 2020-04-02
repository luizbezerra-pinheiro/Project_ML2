# Project - Machine Learning 2

**Authors:** Luiz PINHEIRO and Matheus MARCONDES

## Problem

Create a model that predicts whether we can issue credit to a client or not.

We are given a dataset of the client characteristics and if the company issued credit to them.

## Overview of the problem

### What is the objective?

Increase the profit of the company by issuing credit to a trusted client 

### How will our solution be used?

In summary, our model will be used to detect the trusted clients and 
then make money with them by issuing credit. 

### Type of the problem

Binary classification (Supervised Learning) by using batch learning. 

### How should performance be measured? 
Is the performance aligned with the business objective?**

The objective is to maximize our revenue by issuing credit to the clients. In other 
words, it can be seen as finding the maximum of true positives (trusted clients 
predicted as trusted) with few false positives (untrusted clients predicted as trusted).

***First method (Hyphotetical revenue)*** 

Since we don't have any information about the amount of credit issue that will be 
issued to the clients, we can consider the revenue obtained by the company as:

- 1, for each true positives
- -1, for each false positives
- 0 for the negatives.

If we have, for example, 100 clients predicted as positive whose 20 are not trusted, then
the hypothetical revenue of the company is 80 - 20 = 60, according to our performance 
measure. But, if we have 75 positives whose 5 are not trusted, then the revenue will be
70 - 5 = 65, which is better than the first case.

In order to normalize this value, we will consider the following performance :

<img src="https://render.githubusercontent.com/render/math?math=\dfrac{TP-FP}{TP %2B FN}">


***Second method (ROC curve and AUC)***

We are looking for a high recall and a low false positive. The ROC curve shows the nuances
between the recall and the false positive rate. We can use the AUC (area under the curve 
ROC) as a performance measure. AUC is a good metric for imbalanced classification problems.

### What is the minimum performance?

*First method:* The minimum performance is zero, which means that we can not lose money.

*Second method :* Predicting randomly always produces an AUC
of 0.5, no matter how imbalanced the classes in a dataset are.