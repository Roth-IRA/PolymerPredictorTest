# Package Imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, ChemicalFeatures, rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput
from sklearn.impute import KNNImputer

# Loading Models
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
st.write("Tg: Glass transition temperature (Celsius)")
st.write("Tc: Thermal conductivity (W/m*K)")
st.write("Rg: Radius of gyration")
st.write("FFV: Fractional free volume")
st.write("Density: Polymer density (g*cm^3)")
user_input = st.text_input("SMILES:")

def smiles_to_fp(smiles, radius=4, n_bits = 2048):
    mol = Chem.MolFromSmiles(smiles) # this creates a molecule object from SMILES notation (used to make sure the polymer is valid)
    if mol is not None:
        # SMILES ("Simplified Molecular Input Line Entry System") is notation for describing the structure of a polymer
        # MorganGenerator transforms SMILES from text to numbers
         # The fingerprint is a binary array which identifies which features a polymer does and does not have
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = 2048)
        arr = np.zeros((n_bits,),dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vect,arr)
        return arr
    else:
        return None # Returns None for invalid SMILES

user_input_fp = smiles_to_fp(user_input)

if user_input_fp is not None:
    user_input_fp = user_input_fp.reshape(1,-1)
    
if user_input_fp is not None:
    st.subheader("Predicted Chem Properties")
    
    predictionTg = st.session_state.modelTg.predict(user_input_fp)
    st.write(f"Tg: {predictionTg}")
    
    predictionTc = st.session_state.modelTc.predict(user_input_fp)
    st.write(f"Tc: {predictionTc}")
    
    predictionRg = st.session_state.modelRg.predict(user_input_fp)
    st.write(f"Rg: {predictionRg}")
    
    predictionFFV = st.session_state.modelFFV.predict(user_input_fp)
    st.write(f"FFV: {predictionFFV}")
    
    predictionDensity = st.session_state.modelDensity.predict(user_input_fp)
    st.write(f"Density: {predictionDensity}")
    
else:
    st.write("Please input a rdkit-recognized chemical structure!")

st.subheader("Fingerprint Substructures")
molecule = Chem.MolFromSmiles(user_input)
if molecule is not None:
    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=4, nBits=1024, bitInfo=bitInfo)
        
    for bit_id in fp.GetOnBits(): # The bitInfo parameter enables more interpretability by providing information on which atoms make up each bit/substructure
        if bit_id in bitInfo:
            for atom_idx, radius in bitInfo[bit_id]:
                atom = molecule.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                st.write(f"Bit {bit_id} set by atom {atom_idx} ({symbol}) with radius {radius}")
    
