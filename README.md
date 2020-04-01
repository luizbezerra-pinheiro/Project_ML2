# Project - Machine Learning 2

**Authors:** Luiz PINHEIRO and Matheus MARCONDES

## Problem

Create a model that predicts whether we can issue credit to a client or not.

We are given a dataset of the client characteristics and if the company issued credit to them.

## Overview of the problem

**What is the objective?**

Increase the profit of the company by issuing credit to a trusted client 

**How will our solution be used?**

In summary, our model will be used to detect the trusted clients and 
then make money with them by issuing credit. 

**Type of the problem**

Binary classification (Supervised Learning) by using batch learning. 

**How should performance be measured? 
Is the performance aligned with the business objective?**

The objective is to maximize our revenue by issuing credit to the clients. In other 
words, it can be seen as finding the maximum of true positives (trusted clients 
predicted as trusted) with few false positives (untrusted clients predicted as trusted). 

Since we don't have any information about the quantity of credit issue that will be 
issued to the clients, we can consider the revenue obtained by the company as:

- 1, for each true positives
- -1, for each false positives
- 0 for the negatives.

If we have, for example, 100 clients predicted as positive whose 20 are not trusted, then
the hypothetical revenue of the company is 80 - 20 = 60, according to our performance 
measure. But, if we have 75 positives whose 5 are not trusted, then the revenue will be
70 - 5 = 65, which is better than the first case.

**What is the minimum performance?**

The minimum performance is zero, which means that we can not lose money.