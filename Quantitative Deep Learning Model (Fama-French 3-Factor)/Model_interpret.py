import shap 
from LSTM_arch import ARch
from merged import load_data
import numpy as np
model, history, X_test, y_test, X_train, y_train  = ARch()
model_data = load_data()

print("start SHAP interpretation")

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

explainer = shap.GradientExplainer(model, background)

shap_values = explainer.shap_values(X_test)

shap_values_3d = shap_values

num_features = X_train.shape[2] 
feature_names = model_data.columns[0:num_features]

shap_values_bar= np.mean(np.abs(shap_values_3d), axis=1)
shap_values_beeswarm = np.mean(shap_values_3d, axis=1)

features_2d_for_plot = X_test.mean(axis=1)

print("Generating Global features Importance plot")
shap.summary_plot(shap_values_bar, features =features_2d_for_plot,feature_names = feature_names, plot_type = "bar", show = True)

print("Generating Beeswarm summary plot")
shap.summary_plot(shap_values_beeswarm, X_test, feature_names= feature_names, show= True)