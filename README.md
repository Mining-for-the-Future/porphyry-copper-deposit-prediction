# Porphyry Copper Deposit prediction using ASTER satellite imagery
In response to climate change, the world requires a rapid transition away from fossil fuels toward renewable sources of energy. This transition entails the construction of infrastructure and equipment for the production and storage of electricity on a massive scale. As a result, we need to increase production of certain critical minerals. 

Copper is used in a wide variety of renewable energy technologies, and Porphyry Copper Deposits (PCDs) are an excellent source of this metal. While mining these deposits comes with undeniable environmental impacts, the gains in terms of avoided CO2 emissions are enormous.

>The emissions created by extracting minerals from the ground are tiny compared to those created by burning fossil fuels.[^1]

Identifying new PCDs on a large scale, therefore, could be a major contribution to the fight against climage change.

## The project
Craig Nicolay and Eli Weaverdyck have taken on the task of building a tool that will predict the occurrence of PCDs anywhere in the world as the final project of their WBS CODING SCHOOL Data Science Bootcamp. This repository contains modules and notebooks that:
1. Demonstrate the steps they took to collect training data from Google Earth Engine and train a convolutional neural network to classify ASTER satellite imagery scenes as either containing or not containing a PCD, and
2. Predict the locations of PCDs within a user-provided polygon.

The process of moving from training data to prediction software can be followed by reading the notebooks in the following order:
1. TrainingDataGenerate_final.ipynb
2. ModelTraining_final.ipynb
3. MiningfortheFutureUI_final.ipynb

## A PCD predictor
The PCD predictor can be used entirely from the final notebook, MiningfortheFutureUI_final.ipynb, given a model weights file, which can be generated using the code in ModelTraining_final.ipynb.
The user imports a polygon designating the area of interest.
![screenshot_mine_polygon](https://github.com/Mining-for-the-Future/porphyry-copper-deposit-prediction/assets/81333200/ed3bcdbc-d17d-47b8-94fd-fa74bec4bd24)

The program then divides the polygon into 6 x 6 km squares. These squares overlap each other in both directions so that most locations within the polygon are covered by four squares.
![screenshot_mine_predicting_polygons](https://github.com/Mining-for-the-Future/porphyry-copper-deposit-prediction/assets/81333200/10573ba8-2cfb-45f1-8300-84593080ee92)
ASTER satellite imagery is collected for each square, processed, and passed to the model for prediction.

The model then calculates the average prediction for the 3 x 3 km squares where a unique combination of four larger squares overlap.
![screenshot_mine_results_legend](https://github.com/Mining-for-the-Future/porphyry-copper-deposit-prediction/assets/81333200/bc9d279f-1706-4275-9d48-2cc8924df241)

This polygon was drawn over the Escondida copper mine in Chile. The model predicts that the mine is NOT the location of a PCD because it is trained to detect only unexploited PCDs. It does suggest, however, that there might be further PCDs to the southeast of the current mine!

[^1]: [Ferreira, Fernanda. "How does the environmental impact of mining for clean energy metals compare to mining for coal, oil and gas?" Ask MIT Climate, May 8, 2023.](https://climate.mit.edu/ask-mit/how-does-environmental-impact-mining-clean-energy-metals-compare-mining-coal-oil-and-gas)
