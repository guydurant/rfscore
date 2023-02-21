from oddt.toolkits import ob
from oddt.scoring import descriptors
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
from argparse import ArgumentParser
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    keys = df['key'].values
    protein_files = [os.path.join(data_dir, file) for file in df['protein'].values]
    ligand_files = [os.path.join(data_dir, file) for file in df['ligand'].values]
    pks = df['pk'].values
    return keys, protein_files, ligand_files, pks


def generate_feature(protein_file, ligand_file):
    protein = next(ob.readfile('pdb', protein_file))
    protein.protein = True
    ligand = next(ob.readfile('sdf', ligand_file))
    rfscore_engine = descriptors.close_contacts_descriptor(
        protein=protein,
        cutoff=12,
        ligand_types = [6, 7, 8, 9, 15, 16, 17, 35, 53],
        protein_types = [6, 7, 8, 16],
    )
    return {name:value for name, value in zip(rfscore_engine.titles, rfscore_engine.build(ligand)[0])}

def batch_generate_features(csv_file, data_dir):
    keys, protein_files, ligand_files, pks = load_csv(csv_file, data_dir)
    if not os.path.exists(f'temp_features/{csv_file.split("/")[-1].split(".")[0]}_features.csv'):
        with Parallel(n_jobs=-1) as parallel:
            features = parallel(delayed(generate_feature)(protein_files[i], ligand_files[i]) for i in tqdm(range(len(keys))))
        features_df = pd.DataFrame(features)
        features_df.to_csv(f'temp_features/{csv_file.split("/")[-1].split(".")[0]}_features.csv', index=False)
    else:
        features_df = pd.read_csv(f'temp_features/{csv_file.split("/")[-1].split(".")[0]}_features.csv')
    return features_df, pks, keys
    

def train_model(csv_file, data_dir):
    features_df, pks, keys = batch_generate_features(csv_file, data_dir)
    print('Ready to train model')
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    model.fit(features_df, pks)
    return model

def predict(model, csv_file, data_dir):
    features_df, pks, keys = batch_generate_features(csv_file, data_dir)
    pred_pK = model.predict(features_df)
    return pd.DataFrame({'key': keys, 'pred': pred_pK, 'pk': pks})

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv_file', help='CSV file with protein, ligand and pk data')
    parser.add_argument('--data_dir', help='Directory with protein and ligand files')
    parser.add_argument('--val_csv_file', help='CSV file with protein, ligand and pk data')
    parser.add_argument('--val_data_dir', help='Directory with protein and ligand files')
    parser.add_argument('--model_name', help='Name of the model to be saved or to be loaded')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict pK')
    args = parser.parse_args()
    if args.train:
        model = train_model(args.csv_file, args.data_dir)
        with open(f'temp_models/{args.model_name}.pkl', 'wb') as handle:
            pickle.dump(model, handle)
    elif args.predict:
        with open(f'temp_models/{args.model_name}.pkl', 'rb') as handle:
            model = pickle.load(handle)
        results_df = predict(model, args.val_csv_file, args.val_data_dir)
        results_df.to_csv(f'results/{args.model_name}_{args.val_csv_file.split("/")[-1]}', index=False)
        print(f'Pearson: {pearsonr(results_df["pred"], results_df["true"])[0]}')
        print(f'Spearman: {spearmanr(results_df["pred"], results_df["true"])[0]}')
    else:
        raise ValueError('Need to define mode, --train or --predict')

