
# Deployment_PROJECT(Solar_Radiation_Prediction) By DeepLearning Model
### The `Web Application` I Created found in  <a href="https://radiation-prediction-dl.herokuapp.com/" target="_blank">`This Link`</a>.
#### The Data used here can be found through  <a href="https://www.kaggle.com/dronio/SolarEnergy#" target="_blank">`This Link`</a> on Kaggle Website.

## Description of Data:
#### The dataset contains such columns as: `wind direction`, `wind speed`, `humidity` and `temperature`. 
#### The response parameter that is to be predicted is: `Solar_radiation`. 
#### It contains measurements for the past 4 months and you have to `predict the level of solar radiation`.
### `Just imagine that you've got solar energy batteries and you want to know will it be reasonable to use them in future`.

##  About The Data:
#### These datasets are meteorological data from the `HI-SEAS weather station` from `four months` (September through December 2016) between Mission IV and Mission V.

### For each dataset, the fields are:
- A row number (1-n) useful in sorting this export's results
- The `UNIX time_t` date (seconds since Jan 1, 1970). Useful in sorting this export's results with other export's results
- The date in yyyy-mm-dd format
- The local time of day in hh:mm:ss 24-hour format
- The numeric data, if any (may be an empty string)
- The text data, if any (may be an empty string)

### The Units Of Each Dataset are:
- `Solar radiation`: watts per meter^2

- `Temperature`: degrees Fahrenheit

- `Humidity`: percent

- `Barometric pressure`: Hg

- `Wind direction`: degrees

- `Wind speed`: miles per hour

- `Sunrise/sunset`: Hawaii time

### - `'Solar radiation'`: The target variable. 

## Finally, After creating the `DeepLearning Model` and save as `h5` file, Deploy the model with `FLASK` and upload it to`Heroku Platform`..
## The `Web Application` I Created found in  <a href="https://radiation-prediction-dl.herokuapp.com/" target="_blank">`This Link`</a>.
