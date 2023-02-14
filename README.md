
![logo](frontend/logo.png)

The experience of booking a train with Deutsche Bahn often requires guesswork. At the time of booking, you don't know if a train is usually delayed, or whether other information could help you make an informed decision about which train to choose. We developed a machine learning model to help people make better decisions for their train journeys in Germany.

## How it works

We gathered records of all high-speed train rides between Cologne, Munich and Berlin in 2020 and 2021 and combined it with historical weather information for each ride. Then, we encoded real-world information in a way that could be understood by a machine learning model - for example, we filtered periods with lockdowns, flagged all major holidays and added variables indicating the week day of a given ride, as all these factors could be relevant for the model's performance. Our tests suggest that the model can predict - with about 80% accuracy - whether a given ride between these three cities will be delayed or not.

What is happening behind the scenes here is that we are sending the information input by the user to an API from Deutsche Bahn, which gives us a list of possible journeys on the specified date and time. Then, we send this info to our own API, which gathers weather predictions and combines that with other info we have about your ride, producing a prediction. Due to its dependence on weather, we restrict searches to a period of 15 days from the present.

## Tech Stack

We developed our app on Python and deployed it with Heroku (2023 update: we are now deployed at streamlit.app). Our model - lightGBM, a flavor of xgboost - was trained using the Keras library, deployed at Google Cloud Platform and made interactive via an API. Please check our site here: https://goodtrainbadtrain.streamlit.app/

## About us

This project was developed by [Stefano Pupe](https://github.com/spupe), [Marie Macnee](https://github.com/mariemacnee), [Boris Bohsem](https://github.com/boribo7) and [Juan Cotrino](https://github.com/juancotrino) in a 2-week sprint as a requirement for the completion of a Data Science Bootcamp at Le Wagon. We would like to thank Deutsche Bahn, [meteostat.net](https://meteostat.net/en/) and Statweather for their APIs, Jose Aponte for the logo and [zugfinder.net](https://www.zugfinder.net/en/start) for the train dataset.
