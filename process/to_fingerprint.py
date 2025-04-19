from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import AllChem, MACCSkeys


def molecule_to_maccs(x):
    maccs_ls = []
    maccs = MACCSkeys.GenMACCSKeys(x).ToBitString()
    # maccs_ls.append(int(maccs[bit]) for bit in range(len(maccs)))
    for bit in range(len(maccs)):
        maccs_ls.append(int(maccs[bit]))
    return maccs_ls
    # return MACCSkeys.GenMACCSKeys(x).ToBitString()


def molecule_to_ecfp4(x):
    ecfp4_ls = []
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048).ToBitString()
    # ecfp4_ls.append(int(bit) for bit in ecfp4)
    for bit in range(len(ecfp4)):
        ecfp4_ls.append(int(ecfp4[bit]))
    return ecfp4_ls
    # return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048).ToBitString()


def molecule_to_fcfp4(x):
    fcfp4_ls = []
    fcfp4 = AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True).ToBitString()
    # fcfp4_ls.append(int(bit) for bit in fcfp4)
    for bit in range(len(fcfp4)):
        fcfp4_ls.append(int(fcfp4[bit]))
    return fcfp4_ls
    # return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True).ToBitString()


def molecule_to_fingerprint(smiles_ls):
    data = pd.DataFrame()
    lost = pd.DataFrame()
    maccs = pd.DataFrame(columns=range(167))
    ecfp4 = pd.DataFrame(columns=range(2048))
    fcfp4 = pd.DataFrame(columns=range(2048))
    length, length_ = 0, 0

    for i in tqdm(range(len(smiles_ls))):
        mol = AllChem.MolFromSmiles(file.at[i, 'smiles'])
        if mol is not None:
            data.at[length, 'smiles'] = file.at[i, 'smiles']
            data.at[length, 'type'] = file.at[i, 'type']
            maccs.loc[length] = molecule_to_maccs(mol)
            ecfp4.loc[length] = molecule_to_ecfp4(mol)
            fcfp4.loc[length] = molecule_to_fcfp4(mol)
            length += 1
        else:
            # print(f"{i}:something wrong")
            lost.at[length_, 'smiles'] = file.at[i, 'smiles']
            lost.at[length_, 'type'] = file.at[i, 'type']
            length_ += 1

    data.to_csv("../data/mito_data.csv", index=False)
    maccs.to_csv("../data/mito_maccs.csv", index=False)
    ecfp4.to_csv("../data/mito_ecfp4.csv", index=False)
    fcfp4.to_csv("../data/mito_fcfp4.csv", index=False)

    lost.to_csv("../data/mito_lost.csv", index=False)
    return f'success:{len(data)} wrong:{len(lost)}'


file = pd.read_csv("../data/mito_toxicity_data.csv")
molecule_to_fingerprint(file['smiles'].tolist())


