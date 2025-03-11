

import pandas as pd
import numpy as np
import logging
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer

dfx = pd.read_csv('x_train_final.csv')
dfy=pd.read_csv('y_train_final_j5KGWWK.csv')
df_test=pd.read_csv('x_test_final.csv')

df_test=df_test.drop('Unnamed: 0', axis=1)
dfx=dfx.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
dfy=dfy.drop('Unnamed: 0', axis=1)
dfy['p0q0'] = dfy['p0q0'].astype(int)

dfx= pd.concat([dfx, dfy['p0q0']], axis=1)


# gares_a_supprimer = ["OUA", "TXR", "BKS"]
# dfx = dfx[~dfx['gare'].isin(gares_a_supprimer)]

# # Convert 'date' to datetime
# dfx['date'] = pd.to_datetime(dfx['date'], errors='coerce')
# df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')

# # Extract date features
# dfx['year'] = dfx['date'].dt.year
# dfx['month'] = dfx['date'].dt.month
# dfx['day'] = dfx['date'].dt.day
# dfx['day_of_week'] = dfx['date'].dt.dayofweek

# df_test['year'] = df_test['date'].dt.year
# df_test['month'] = df_test['date'].dt.month
# df_test['day'] = df_test['date'].dt.day
# df_test['day_of_week'] = df_test['date'].dt.dayofweek

# # Drop 'date' column after extracting features
# df_test = df_test.drop(columns=['date'])
# dfx = dfx.drop(columns=['date'])



# Drop rows with NaN values
df_test = df_test.dropna()
dfx = dfx.dropna()

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['arret', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
dfx[numerical_features] = scaler.fit_transform(dfx[numerical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])

"""# Model training"""

ray.shutdown()
ray.init(num_cpus=107)

"""# Modèle 1 : MAE=0.65"""




# Create the scorer using make_scorer
mae_metric = make_scorer(name='mae_class', score_func=sklearn.metrics.mean_absolute_error, optimum=0,greater_is_better=False, needs_class=True)


predictor = TabularPredictor(
    label='p0q0',
    eval_metric=mae_metric,
    problem_type='multiclass'
).fit(
    time_limit=10*3600,
    train_data=dfx,
    presets='best_quality'
)




# Supposons que 'predictor' et 'df_test' sont déjà définis
preds = predictor.predict(df_test)
preds = pd.DataFrame(preds)

predictions_df = pd.DataFrame({
    'Unnamed: 0': df_test.index,  # Utilisation des indices de df_test comme identifiants
    'p0q0': preds['p0q0']  # Ajouter les prédictions dans la colonne p0q0
})

# Sauvegarder les prédictions dans un fichier CSV
predictions_df.to_csv('predictionnuit.csv', index=False)