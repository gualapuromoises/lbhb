#! /usr/bin/env python3

import math
import pandas as pd
import sys
import numpy as np
from Bio.PDB import *

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth',None)

# Folders and files for processing 
srt_i =0#srt_i=int(sys.argv[1])
stp_i = 999999 #stp_i=int(sys.argv[2])
data_dir = "../data"
with open("../data/pdb_list.txt", encoding="utf-8") as f:
    read_data = f.read()
    pdb_list = read_data.split('\n')

print("number of pdbs = %s"%len(pdb_list))

if stp_i < len(pdb_list):
    pdb_list = pdb_list[srt_i:stp_i]
else:
    pdb_list = pdb_list[srt_i:]
    stp_i=len(pdb_list)
print("number of pdbs = %s"%len(pdb_list))
print(pdb_list[:10])

MAX_LBHB_DIST = 6.5
METAL_RES_CODES= [ "FE" ,"FE2","FES","FEO"
                 ,"CU" ,"CU1","CUA"
                 ,"MG" ,"ZN" ,"MN"
                 ,"MO" ,"MOO","MOS"
                 ,"NI" ,"3CO","CO"]

# Extract residues, atoms and positions from  PDB files
def split_ATOM(raw):
    return(raw.str[6:11].str.replace(' ', ''),  
           raw.str[11:16].str.replace(' ', ''),
           raw.str[16].str.replace(' ', ''),
           raw.str[17:20].str.replace(' ', ''), 
           raw.str[21].str.replace(' ', ''),
           raw.str[22:26].str.replace(' ', ''),
           raw.str[27].str.replace(' ', ''),
           raw.str[30:37].str.replace(' ', ''),
           raw.str[38:45].str.replace(' ', ''),
           raw.str[46:53].str.replace(' ', ''),
           raw.str[54:59].str.replace(' ', ''),
           raw.str[60:65].str.replace(' ', ''),
           raw.str[72:75].str.replace(' ', ''),
           raw.str[76:78].str.replace(' ', ''),
           raw.str[79:].str.replace(' ', ''))

def get_ATOM_DF(ATOM, pdb=None):
    atom_data = split_ATOM(ATOM['raw'])
    ATOM = pd.DataFrame({
         'serial':atom_data[0],
         'atom_name':atom_data[1],
         'altLoc':atom_data[2],
         'resName':atom_data[3],
         'chainID':atom_data[4],
         'seqNum':atom_data[5],
         'iCode':atom_data[6],
         'x':atom_data[7],
         'y':atom_data[8],
         'z':atom_data[9],
         'occupancy':atom_data[10],
         'tempFactor':atom_data[11],
         'segID':atom_data[12],
         'element':atom_data[13],
         'charge':atom_data[14] })
    # change coordinates to float to make them usable
    ATOM['x']=ATOM['x'].astype(float)
    ATOM['y']=ATOM['y'].astype(float)
    ATOM['z']=ATOM['z'].astype(float)
    return(ATOM)

def get_struc_atom_coords(fullpath2pdbfile):
    f = open(fullpath2pdbfile, "r")
    pdb_lines = f.readlines()
    pdb_file_df = pd.DataFrame(pdb_lines, columns = ['raw'])
    pdb_file_df['raw'] = pdb_file_df['raw'].str[:80]
    pdb_file_df['key'] = pdb_file_df['raw'].str[:6].str.replace(' ', '')
    ## only keep first model, mostly for NMR models
    atom_coords = get_ATOM_DF(pdb_file_df[pdb_file_df['key'] == 'ATOM'])
    if len(pdb_file_df[pdb_file_df['key'] == 'HETATM'])>0:
        hetatm_coords = get_ATOM_DF(pdb_file_df[pdb_file_df['key'] == 'HETATM'])
        atom_coords = pd.concat([atom_coords, hetatm_coords], ignore_index=True)
    #atom_coords.reset_index(drop=True, inplace=True)
    return(atom_coords)


# Extract pKa atom positions
POTENTIAL_IONIZE_RES_ATOMS = [
    "ARG_NH1", "ARG_NH2", #"ARG_NE",
    "ASP_OD1", "ASP_OD2",
    "GLU_OE1", "GLU_OE2",
    "TYR_OH",
    "HIS_NE2", "HIS_ND1",
    "LYS_NZ", 'LYS_N', 'LYS_O'
    "CYS_SG"  ]

pka_file_cols = ["atom_name", "resName", "seqNum", "predicted_pKa",  "reference_pKa", 
                 "pKa_shift_desolvation_self-energy", "pKa_shift_background_interactions", 
                 "pKa_shift_charge-charge_interactions", "generalized_Born_radius", 
                 "solvent_exposure_parameter"]

def get_resi_pKas(pka_file):
    pka_data = pd.read_csv(pka_file, sep='\s+', names=pka_file_cols, skiprows=1)
    pka_data = pka_data.loc[~pka_data['atom_name'].isin(["N", "C"])]
    return(pka_data)

def get_pka_coords_data(this_pka_data, this_atom_coords, this_struc_id):
     for this_df in [this_pka_data, this_atom_coords]:
          this_df["atom_name"]=this_df["atom_name"].astype(str).str[:]
          this_df["resName"]=this_df["resName"].astype(str)
          this_df["seqNum"]=this_df["seqNum"].astype(int)

     del this_pka_data['atom_name']
     this_pka_coords_data = pd.merge(this_pka_data, this_atom_coords, 
                                     left_on=["resName", "seqNum"], 
                                     right_on=["resName", "seqNum"] )
     
     ## only keep atoms that could form a sidechain LBHB
     this_pka_coords_data['res_atom'] = (
          this_pka_coords_data['resName']+"_"+this_pka_coords_data['atom_name'])
     this_pka_coords_data = this_pka_coords_data.loc[
          this_pka_coords_data['res_atom'].isin(POTENTIAL_IONIZE_RES_ATOMS)]
     this_pka_coords_data.reset_index(drop=True, inplace=True)
     return(this_pka_coords_data)

# Obtain solvation energy values for each atom
def solvnrg(slvnrg_file):
     with open(slvnrg_file, 'r') as f:
          data = f.read()
     lines = data.split('\n')
     df = pd.DataFrame([line.split() for line in lines if line.startswith('SOLV')],
                    columns=['c1', 'c2', 'c3', 'atom', 'res', 'resn', 'charge', 'solv_nrg'])
     df['atom_code'] = df['res'] + df['resn'] + '_' + df['atom']
     df['charge'] = df['charge'].astype(float)
     return df[['atom_code', 'charge']]
solvnrg_path = '../data/7/a/7adh_A/7adh_A_bluues.solv_nrg'
print(solvnrg(solvnrg_path).shape, solvnrg(solvnrg_path)[:2])

# Get pair of residues, atom positions participating in side-chain Hydrogen bond
def get_LBHB_pairs(this_data_dir, this_struc_id, lbhb_maxdist, metal_maxdist):
    print(this_struc_id)
    this_struc_file = "%s/%s_Rlx.pdb"%(this_data_dir, this_struc_id)
    slvnrg_file = "%s/%s_bluues.solv_nrg"%(this_data_dir, this_struc_id)

    # data from pdb and from solvation_energy
    df_pdb = get_struc_atom_coords(this_struc_file)
    df_pdb['atom_code'] = df_pdb['resName'] + df_pdb['seqNum'] + '_' + df_pdb['atom_name']
    df_slvnrg = solvnrg(slvnrg_file)
    #merge charge values
    df_pdb = df_pdb.merge(df_slvnrg.set_index('atom_code'), how='left', on='atom_code', suffixes=('', '_y'))
    df_pdb = df_pdb.drop('charge', axis=1)
    df_pdb = df_pdb.rename(columns={'charge_y': 'charge'})
    #Select only the atoms in HB 
    NandO_df = df_pdb.loc[df_pdb['element'].isin(['N','O', 'S'])].copy()
    for col in ["atom_name", "resName", "seqNum", "x", "y", "z", 'charge']:
        NandO_df.rename(columns={col:"%s2"%col}, inplace=True)
    ## generate atom pairs
    combinations = []
    temp_df = NandO_df[["atom_name2", "resName2", "seqNum2", "x2", "y2", "z2", "charge2"]]
    temp_df['res_id'] = temp_df["resName2"].astype(str) + temp_df["seqNum2"].astype(str)
    
    new_colnames = ["atom2", "residue2", "resNum2", "x2", "y2", "z2", 'charge2', 'res_id2',\
                    "atom1", "residue1", "resNum1", "x1", "y1", "z1", 'charge1', 'res_id1']
    
    for row1 in temp_df.itertuples():
        for row2 in temp_df.itertuples():
            if row1[8] == row2[8]: #evaluates if residues are the same
                continue
            new_row = row1[1:] + row2[1:]
            combinations.append(new_row)
    atom_pairs = pd.DataFrame(combinations, columns=new_colnames)
    atom_pairs["da_atom2"] = np.where(atom_pairs["charge2"].astype(float) < atom_pairs["charge1"].astype(float), "a", "d")
    atom_pairs["da_atom1"] = np.where(atom_pairs["da_atom2"] == 'a', "d", "a")

    # Caluclate distance between atom pairs 
    atom_pairs = atom_pairs[atom_pairs['res_id2'] != atom_pairs['res_id1']]
    sq_difx = (atom_pairs['x1']-atom_pairs['x2'])**2
    sq_dify = (atom_pairs['y1']-atom_pairs['y2'])**2
    sq_difz = (atom_pairs['z1']-atom_pairs['z2'])**2
    atom_pairs['distance']= (sq_difx + sq_dify + sq_difz).apply(math.sqrt)

    ## bond type  
    atom_pairs['bond_id'] = (atom_pairs["residue1"].astype(str) + "_" + atom_pairs["resNum1"].astype(str) + "_" +
                             atom_pairs["residue2"].astype(str) + "_" + atom_pairs["resNum2"].astype(str))
    
    atom_pairs.loc[atom_pairs["resNum1"]>atom_pairs["resNum2"],'bond_id'] = (
        atom_pairs["residue2"].astype(str) + "_" + atom_pairs["resNum2"].astype(str) + "_" +
        atom_pairs["residue1"].astype(str) + "_" + atom_pairs["resNum1"].astype(str))
    
    # Filter for the shortest
    atom_pairs.sort_values("distance", ascending=True, inplace=True)
    atom_pairs.drop_duplicates(subset=['bond_id'], keep='first', inplace=True, ignore_index=False)
    LBHB_data = atom_pairs.loc[atom_pairs['distance']<= lbhb_maxdist].copy().reset_index(drop=True)


    LBHB_data.loc[LBHB_data['atom1'].isin(['N','O']), 'bond_id'] += '_bb'
    LBHB_data.loc[~LBHB_data['atom1'].isin(['N','O']), 'bond_id'] += '_sc'
    LBHB_data.loc[LBHB_data['atom2'].isin(['N','O']), 'bond_id'] += '_bb'
    LBHB_data.loc[~LBHB_data['atom2'].isin(['N','O']), 'bond_id'] += '_sc'

    for coord_axis in ['x', 'y', 'z']:
        LBHB_data[coord_axis]=LBHB_data["%s1"%(coord_axis)]+LBHB_data["%s2"%(coord_axis)]
        LBHB_data[coord_axis]=LBHB_data[coord_axis]/2

    LBHB_data['metal_name'] = ""
    LBHB_data['metal_num'] = 0
    LBHB_data['metal_dist'] = 10000

    metals = df_pdb.loc[df_pdb["resName"].isin(METAL_RES_CODES)]
    if len(metals)>0:
        for index, row in LBHB_data.iterrows():
            cur_metals = metals.copy()
            cur_metals['distance']=pd.to_numeric(((cur_metals['x'].astype(float)-row['x'])**2 + (cur_metals['y'].astype(float)-row['y'])**2 + \
                                    ((cur_metals['z'].astype(float)-row['z'])**2)).apply(math.sqrt)).round(3)
            min_dist = cur_metals['distance'].min()
            LBHB_data.loc[index, 'metal_dist'] = min_dist
            LBHB_data.loc[index, 'metal_name']= cur_metals.loc[cur_metals['distance']==min_dist, 'resName'].tolist()[0]
            LBHB_data.loc[index, 'metal_num']= cur_metals.loc[cur_metals['distance']==min_dist, 'seqNum'].tolist()[0]
            LBHB_df = LBHB_data.loc[LBHB_data['metal_dist'] <= metal_maxdist].copy().reset_index(drop=True)

    LBHB_df['hb_type'] = LBHB_df['bond_id'].str[-5:]
    LBHB_df['charge_diff'] = LBHB_df['charge2'] - LBHB_df['charge1']
    LBHB_df.loc[LBHB_df["hb_type"] == "sc_bb", "hb_type"] = "bb_sc"
    LBHB_df['distance'] = pd.to_numeric(LBHB_df['distance'], errors='coerce').round(3)
    # Final output data
    LBHB_df = LBHB_df[["atom2", "atom1", "residue2", "residue1","resNum2", "resNum1",
                       'da_atom2', 'distance', 'charge_diff', 'metal_name', 'metal_dist', 'hb_type']]
    print(this_struc_id, LBHB_df.shape)
    return(LBHB_df)

# Run for all files in a folder
def get_LBHB_pairs_for_struc_list(data_directory, struc_list):
    all_LBHB_data = []
    for struc in struc_list:
        try:
          struc_LBHB_data = get_LBHB_pairs("%s/%s/%s/%s"%(data_directory, struc[0], struc[1], struc), struc, 4.5, 15)
          struc_LBHB_data['struc']=struc
          all_LBHB_data.append(struc_LBHB_data)
        except:
            print("failed to add LBHB for %s"%(struc))
    all_LBHB_pairs = pd.concat(all_LBHB_data, ignore_index=True)
    return(all_LBHB_pairs)


data_directory = '../data'
struc_list = pdb_list[:len(pdb_list)-1] # 22268
allprots2 = get_LBHB_pairs_for_struc_list(data_directory, struc_list)
allprots2.to_csv("../results/hb_data2.csv")
print(len(allprots2))