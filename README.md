# CentraleSupélec Summer School on AI 2022 - Multiclass Classification Kaggle Competition
### Classify an email into eight classes based on the metadata extracted from the emails.

This repository contains the work of the WNN team for the Kaggle Challenge of the CentraleSupélec Summer School on AI 2022.

## Description of the competition:

We often face the problem of searching meaningful emails from thousands of promotional emails. This challenge focuses on creating a multi-class classifier that can classify an email into eight classes based on the metadata extracted from the email.

## Results

The used metric was the F1-Score. Our in-competition score was 0.57486. After the competition ended, the following model used by the winning team was tested, and a 0.75602 score was obtained.

```py
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

level0 = list()
level0.append(('lr', LogisticRegression() ))
level0.append(('rf', DecisionTreeClassifier()))
level0.append(('xgb', XGBClassifier()))
level0.append(('lgbm', LGBMClassifier()))

level1 = CatBoostClassifier()
clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=4)

clf.fit(final_train_df, final_train_labels)
pred_y = clf.predict(test_final_df)
```

## Resources

[Link for the Kaggle Competition](https://www.kaggle.com/competitions/centralesupelec-summer-school-on-ai-2022/overview)

[Summer School Webpage](https://www.summerschoolcentralesupelec.fr/artificial-intelligence/)
