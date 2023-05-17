<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img alt="Logo" src="images/pushback_image.jpg" width="800" height="300">
  <h3 align="center"> NASA Pushback competition 2023: Federated Learning </h3>

  <p align="center">
    CDS team repository to develop the models and serve the prediction functionalities for the 2023 NASA Pushback challenge 
    <br />
    <a href="https://www.drivendata.org/competitions/149/competition-nasa-airport-pushback/page/676/"><strong>Visit competition site »</strong></a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
## Table of contents
  <ol>
    <li> <a href="#context-of-federated-learning">Context of Federated Learning</a> </li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#takeaways">Takeaways</a></li>
  </ol>



<!-- CONTEXT OF FEDERATED LEARNING -->
## Context of federated learning

<div align="justify">

Every year millions of flights arrive and depart from US airports. Intelligent scheduling and system control is a very pressing problem. Knowing in advance or accurately forecasting the pushback moment for departure is one of the key inputs to arrange optimized schedules. With the surge of Big Data and Advanced Analytics techniques most of these forecasts are based on Machine Learning methods.

In this 2023 analytics case competition, NASA partnered with DrivenData to facilitate real-world data to develop these types of models and suggest alternative approaches to the existing research.

More precisely, participants in this challenge were given the task to build an Advanced Analytics model capable of forecasting the minutes until pushback at an airplane and timestamp level for 10 US airports.

For a detail on the context of the challenge, approach followed, and results obtained, please visit either the competition site [here](https://www.drivendata.org/competitions/149/competition-nasa-airport-pushback/page/676/) or our Phase I solution repository [here](https://github.com/alsaco33/NASA-pushback).

This repository is concerned with tackling Phase II of the challenge which consists on federating the models developed in Phase I. Since some variables are sensitive and cannot be shared across airlines, a pressing problem is to build accurate models leveraging as much data as possible without allowing airlines to leak information from each other. Some real-world variables which airlines might not want to share with each other are the number of passengers or luggage checked in their flights, but at the same time these are potentially good predictors of pushback time. In this particular Phase II of the combination, two data sources (MFS, and Standtimes) have been obfuscated in order for us to illustrate how a Federated Learning approach could be built.  

There is a plethora of potential methods to be utilized in Federated Learning with varying levels of encryption, sophistication, and specifications based on the use-case requirements. In this study, we will utilize the [Flower framework](https://flower.dev/docs/) with CatBoostRegressor models to build a centralized aggregator which performs vertical federated learning and only shares model weights across clients (in this case airlines). 

<!-- REPOSITORY -->
## Repository structure

Below one can observe the repository structure of the federated learning solution. 

```
PushbackFederated
│   README.md
└───images
└───requirements.txt
└───data
│    └───raw
│           └───KATL
│               └───private
│               └───public
│               └───phase2_train_labels_KATL.csv.bz2
│           └───KCLT
│           └───KDEN
│           └───KDFW
│           └───KJFK
│           └───KMEM
│           └───KMIA
│           └───KORD
│           └───KPHX
│           └───KSEA
│           └───submission_format.csv
│    └───interim
│    └───processed
└───models
│     └───KATL_model
│     └───KCLT_model
│     └───KDEN_model
│     └───KDFW_model
│     └───KJFK_model
│     └───KMEM_model
│     └───KMIA_model
│     └───KORD_model
│     └───KPHX_model
│     └───KSEA_model
└───code
│   └───config.py
│   └───utilities.py
│   └───train.py
│   └───predict.py
```

<!-- USAGE -->
## Usage

In this section we discuss how to use the repository to build the model artifacts, and generate predictions for new, unseen data. There are two main functionalities, *train.py* and *predict.py* which trigger the Federated Learnign training procedure for a given airport and the prediction for the entire submission_format preview respectively. If the raw data is in the folder structure specified above, the packages from *requirements.txt* are installed, these can be triggered through the command line as below:

```python 
$ python3.10 train.py KATL 
```

To train the KATL model setting as clients the different airlines leveraging the Flower framework. And:

```python 
$ python3.10 predict.py
```

In order to retrieve predictions once all the model artifacts have been generated and stored in */models*.

<!-- Takeaways -->
## Takeaways

Through our development and implementation of the models created for phase I in a federated fashion for phase II, we have extracted several takeaways from this technique:

1. Utilizing Flower we have been able to federate the supervised models developed in Phase I of the competition
2. Based on validation MAE performance at an airline level, the results are worse but comparably on par to our Phase I solution
3. There is a significant degree of randomness associated with using Flower in a vertical federated learning setting with boosted trees - due to the relevance of the airline with which we start building our regression trees, and the aleatoric nature of Flower, executing our pipeline multiple times will yield slightly different model artifacts 
4. Off-the-shelf flower implementation lacks a clear guidance on how to define convergence and stop training
5. While illustrative, our method has no guarantees to disable variable leakage, i.e. a bad agent could extract some MFS or Standtime information given the model artifacts, and a more sophisticated approach would need to be put in place in order to provide more strict guardrails 

</div>
