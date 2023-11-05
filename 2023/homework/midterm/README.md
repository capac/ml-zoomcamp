# ML Zoomcamp 2023 &ndash; Midterm Project

The _Heart failure clinical records_ dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

The research article based on the data set states that a random forest model is the best model that can predict survival of patients with heart failure from the serum creatinine and ejection fraction features alone. This statement can already be observed from the feature importance plot where the most important features after follow-up time are serum creatinine and ejection fraction. Moreover, one can also notice the same oberservation in both the exploratory data analysis box plots of death event versus serum creatinine and ejection fraction. Although interesting in itself, further analyses from the exploratory data analysis observations are still warranted to obtain a complete picture of the data set.

The research article can be accessed at the link, [Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone](https://www.semanticscholar.org/paper/Machine-learning-can-predict-survival-of-patients-Chicco-Jurman/e64579d8593140396b518682bb3a47ba246684eb) (link to pdf of article [here](https://bmcmedinformdecismak.biomedcentral.com/counter/pdf/10.1186/s12911-020-1023-5.pdf)).

The data set can be retrieved from the UCI Machine Learning Repository at the link, [Heart failure clinical records](http://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).

The data analysis is organized as follows:

* `notebook.ipynb` contains:
    * Data preparation and data clearning,
    * Exploratory data analysis with box and bar plots,
    * Model selection process and hyper-parameter tuning,
    * Feature importance analysis plot with best model parameters.

* `train.py`
    * Training the final model with best model parameters,
    * Saving best model and DictVectorizer class instance to pickle files.

* `predict.py`
    * Loading the model and DictVectorizer object from the `model.pkl` and `dv.pkl` files respectively,
    * Serving it via Flask web service.
