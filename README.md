[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

# Pushback to the Future: Predict Pushback Time at US Airports

## Goal of the Competition

Coordinating our nation’s airways is the role of the National Airspace System (NAS). The NAS is arguably the most complex transportation system in the world. Operational changes can save or cost airlines, taxpayers, consumers, and the economy at large thousands to millions of dollars on a regular basis. It is critical that decisions to change procedures are done with as much lead time and certainty as possible. The NAS is investing in new ways to bring vast amounts of data together with state-of-the-art machine learning to improve air travel for everyone.

In order to optimize commercial aircraft flights, air traffic management systems need to be able to predict as many details about a flight as possible. One significant source of uncertainty comes right at the beginning of a flight: the pushback time. A more accurate pushback time can lead to better predictability of take off time from the runway. Predicting pushback time depends upon factors like passenger loading, cargo loading, weather, aircraft type, and operator procedures. While available data can be used to improve these predictions, the combination of public and private sources can make it difficult to get access to all of the information needed to make the best predictions. Federated learning (FL) offers immense promise here as an approach to training central ML models using private data held by separate organizations.

This competition involved two phases. In Phase 1, the task was to train a machine learning model to automatically predict pushback time from public air traffic and weather data. Better algorithms for predicting pushback time can help air traffic management systems more efficiently use the limited capacity of airports, runways and the National Airspace System. In Phase 2, we invited the top 5 finalists from Phase 1 to work with NASA to train a federated version of their Phase 1 models.

## What's in this Repository

This repository contains Phase 1 and 2 code from winning competitors in the [Pushback to the Future: Predict Pushback Time at US Airports](https://www.drivendata.org/competitions/group/competition-nasa-airport-pushback/) DrivenData challenge. In Phase 1, teams competed to build the best model to predict pushback time. In Phase 2, the top five teams from Phase 1 experimented with federated learning versions of their Phase 1 models.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User  | Phase 1 Score †  | Phase 2 Score † | Summary of Phase 1 Model | Summary of Phase 2 Model |
----- | ------------- | ---------------- | --------------- | ------------------------ | ------------------------ |
1     | Team CDS      | 10.673           | 16.329          | Ensemble multiple CatBoost regressors per airport. | One CatBoost per airport; models trained in sequence across airlines. |
2     | Moles         | 10.728           | 36.589          | One CatBoost regressor per airport. | [FedXgbNnAvg](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedxgb_nn_avg.py) approach; first train one 50-tree CatBoost regressor per airline, use the concatenated tree outputs for all airlines as features to a 1D convolutional neural network, which is trained in a federated fashion using federated weight averaging. |
3     | Oracle 2      | 11.054           | 12.848          | Multi-stage pipeline (one per airport); XGBoost regressor followed by XGBoost classifier that detects and adjusts under- or over-estimates. | Ensemble of XGBoost model trained on public features and federated neural network model trained with federated weight averaging on private and public features. |
4     | FLHuskies2    | 11.105           | 104.274         | One LightGBM regressor per airport. | Federated multilayer perceptron trained with federated weight averaging. |
5     | Cuong_Syr     | 11.946           | 14.519          | One XGBoost regressor per airport. | Federated multilayer perceptron trained with federated weight averaging. |

† Scores are mean absolute error in minutes.

Additional solution details can be found in the `reports` folder inside the directory for each submission.


## Additional resources

- [Benchmark blog post](https://drivendata.co/blog/airport-pushback-benchmark)
- [Winner's blog post](https://drivendata.co/blog/airport-pushback-finalists)
