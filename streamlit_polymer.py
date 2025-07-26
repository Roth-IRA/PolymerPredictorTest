import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor # opted for XGBoost since it routinely outperforms most other algorithms in kaggle competitions
from sklearn.metrics import mean_absolute_error # the competition uses a weighted MAE since it involves making predictions for multiple numeric variables
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from sklearn.impute import KNNImputer

modelTg = XGBRegressor()
modelTc = XGBRegressor()
modelRg = XGBRegressor()
modelFFV = XGBRegressor()
modelDensity = XGBRegressor()
modelTg = modelTg.load_model("modelTg.bin")
modelTc = modelTc.load_model("modelTc.bin")
modelRg = modelRg.load_model("modelRg.bin")
modelFFV = modelFFV.load_model("modelFFV.bin")
modelDensity = modelDensity.load_model("modelDensity.bin")

st.title("Polymer Prediction, Carl R.")

st.write("### Input valid SMILES notation for a chemical species to get chemical property predictions!")
user_input = st.text_input("SMILES:")

def smiles_to_fp(smiles, radius=4):
    mol = Chem.MolFromSmiles(smiles) # this creates a molecule object from SMILES notation (used to make sure the polymer is valid)
    if mol is not None:
        # SMILES ("Simplified Molecular Input Line Entry System") is notation for describing the structure of a polymer
        # MorganGenerator transforms SMILES from text to numbers
        generator = AllChem.GetMorganGenerator(radius=radius)
        return np.array(generator.GetFingerprint(mol))
    else:
        return None # Returns None for invalid SMILES

user_input_fp = smiles_to_fp(user_input)
if user_input_fp is not None:
    st.write("Predicted Chem Properties: ")
    
    predictionTg = modelTg.predict(user_input_fp)
    st.write("Tg: ", predictionTg)
    
    predictionTc = modelTc.predict(user_input_fp)
    st.write("Tc: ", predictionTc)
    
    predictionRg = modelRg.predict(user_input_fp)
    st.write("Rg: ", predictionRg)
    
    predictionFFV = modelFFV.predict(user_input_fp)
    st.write("Tg: ", predictionFFV)
    
    predictionDensity = modelDensity.predict(user_input_fp)
    st.write("Tg: ", predictionDensity)
    
else:
    st.write("Please input a rdkit-recognized chemical structure!")
