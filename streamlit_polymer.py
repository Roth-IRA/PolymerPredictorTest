import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor # opted for XGBoost since it routinely outperforms most other algorithms in kaggle competitions
from sklearn.metrics import mean_absolute_error # the competition uses a weighted MAE since it involves making predictions for multiple numeric variables
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from sklearn.impute import KNNImputer

if "modelTg" not in st.session_state:
    st.session_state.modelTg = xgb.XGBRegressor()
    st.session_state.modelTg.load_model("modelTg.bin")

if "modelTc" not in st.session_state:
    st.session_state.modelTc = xgb.XGBRegressor()
    st.session_state.modelTc.load_model("modelTc.bin")

if "modelRg" not in st.session_state:
    st.session_state.modelRg = xgb.XGBRegressor()
    st.session_state.modelRg.load_model("modelRg.bin")

if "modelFFV" not in st.session_state:
    st.session_state.modelFFV = xgb.XGBRegressor()
    st.session_state.modelFFV.load_model("modelFFV.bin")

if "modelDensity" not in st.session_state:
    st.session_state.modelDensity = xgb.XGBRegressor()
    st.session_state.modelDensity.load_model("modelDensity.bin")

st.title("Polymer Prediction, Carl R.")

st.write("### Input valid SMILES notation for a chemical species to get chemical property predictions!")
user_input = st.text_input("SMILES:")

def smiles_to_fp(smiles, radius=4, n_bits = 2048):
    mol = Chem.MolFromSmiles(smiles) # this creates a molecule object from SMILES notation (used to make sure the polymer is valid)
    if mol is not None:
        # SMILES ("Simplified Molecular Input Line Entry System") is notation for describing the structure of a polymer
        # MorganGenerator transforms SMILES from text to numbers
         # The fingerprint is a binary array which identifies which features a polymer does and does not have
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = 2048)
        arr = np.zeros((n_bits,),dtype=np.unit8)
        DataStructs.ConvertToNumpyArray(bit_vect,arr)
        return arr
    else:
        return None # Returns None for invalid SMILES

user_input_fp = smiles_to_fp(user_input)

if user_input_fp is not None:
    user_input_fp = user_input_fp.reshape(1,-1)
    
if st.button("Predict Properties") and user_input_fp is not None:
    st.write("Predicted Chem Properties: ")
    
    predictionTg = st.session_state.modelTg.predict(user_input_fp)
    st.write("Tg: ", predictionTg)
    
    predictionTc = st.session_state.modelTc.predict(user_input_fp)
    st.write("Tc: ", predictionTc)
    
    predictionRg = st.session_state.modelRg.predict(user_input_fp)
    st.write("Rg: ", predictionRg)
    
    predictionFFV = st.session_state.modelFFV.predict(user_input_fp)
    st.write("Tg: ", predictionFFV)
    
    predictionDensity = st.session_state.modelDensity.predict(user_input_fp)
    st.write("Tg: ", predictionDensity)
    
else:
    st.write("Please input a rdkit-recognized chemical structure!")
