import joblib
from preprocessing import *

numMod = input("Entrez le numéro de modèle que vous voulez créer ? \n1 : CountVectorizer & MultinomialNB \n2 : TfidfVectorizer & MultinomialNB \n3 : CountVectorizer & Logistic Regression \n4 : TfidfVectorizer & Logistic Regression \n")

print(numMod)
# load the model
gs1=joblib.load("model_gs.pkl")
Xcvec_df_a1=preprocess("Russia deploys trained dolphins to guard Black Sea naval base")
pred = gs1.predict(Xcvec_df_a1)
print("-------------------")
print(pred)