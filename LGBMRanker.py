import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

gbm = lgb.LGBMRanker()
data=pd.read_csv("/home/yavuz/yavuz_proje/allosteric_feature_selected/data/training_data/training_ready.csv")
fpocket_features=['Score', 'Druggability Score', 'Number of Alpha Spheres', 'Total SASA', 'Polar SASA', 'Apolar SASA', 'Volume',
        'Mean local hydrophobic density', 'Mean alpha sphere radius', 'Mean alp. sph. solvent access',
        'Apolar alpha sphere proportion', 'Hydrophobicity score', 'Volume score',
        'Polarity score', 'Charge score', 'Proportion of polar atoms', 'Alpha sphere density',
        'Cent. of mass - Alpha Sphere max dist', 'Flexibility',]
X=data.loc[:, fpocket_features]
y=data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


query_train = [X_train.shape[0]]
query_val = [X_val.shape[0]]
query_test = [X_test.shape[0]]

gbm.fit(X_train, y_train, group=query_train,
        eval_set=[(X_val, y_val)], eval_group=[query_val],
        eval_at=[5, 10, 20], )

test_pred = gbm.predict(X_test)
X_test["predicted_ranking"] = test_pred
X_test.sort_values("predicted_ranking", ascending=False)