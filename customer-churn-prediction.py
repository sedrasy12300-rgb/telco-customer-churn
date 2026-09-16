import pandas as pd
from sklearn.model_selection import  train_test_split ,GridSearchCV ,cross_val_score ,cross_validate

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score ,confusion_matrix ,classification_report 

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
import joblib

df =pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


df =df.drop("customerID",axis=1)
df["Churn"]=df["Churn"].map({"Yes":1,"No":0})
df["TotalCharges"] =pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"] =df["TotalCharges"].fillna(df["TotalCharges"].mean())
text_columns =df.select_dtypes(include="object").columns
df =pd.get_dummies(df,columns=text_columns)

x =df.drop("Churn",axis=1)
y =df["Churn"]
x_train,x_test ,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model= RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

param ={"n_estimators":[100,200,300],"max_depth":[3,5,10,None],"class_weight":["balanced"]}
grid =GridSearchCV(RandomForestClassifier(random_state=42),param_grid=param,cv=5,scoring="recall")
grid.fit(x_train,y_train)
print(grid.best_estimator_)
print(grid.best_score_)

best_model = grid.best_estimator_

y_grid = best_model.predict(x_test)

print(accuracy_score(y_test,y_grid))
print(confusion_matrix(y_test,y_grid))
print(classification_report(y_test,y_grid))

pipe =Pipeline([("scaler" ,StandardScaler()),("model",LogisticRegression(random_state=42))])
pipe.fit(x_train,y_train)
y_pipe =pipe.predict(x_test)
print(accuracy_score(y_test,y_pipe))
print(confusion_matrix(y_test,y_pipe))
score =cross_val_score(best_model,x,y,cv=5)
score_val = cross_validate(best_model,x,y,cv=5,scoring=["accuracy","recall","precision","f1"])
print(score.mean())
print(score_val)

importance =pd.DataFrame({"Feature":x.columns,"Importance": best_model.feature_importances_})

importance =importance.sort_values(by="Importance",ascending=False)
print(importance)
joblib.dump(best_model,"telco_churn_model.pkl")
