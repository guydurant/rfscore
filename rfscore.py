from oddt.toolkits import ob
from oddt.scoring import descriptors
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
from argparse import ArgumentParser
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from joblib import Parallel, delayed

def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    keys = df['key'].values
    protein_files = [os.path.join(data_dir, file) for file in df['protein'].values]
    ligand_files = [os.path.join(data_dir, file) for file in df['ligand'].values]
    pks = df['pk'].values
    return keys, protein_files, ligand_files, pks


def generate_features(protein_file, ligand_file):
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

def train_model(csv_file, data_dir):
    print('Featurising data')
    keys, protein_files, ligand_files, pks = load_csv(csv_file, data_dir)
    with Parallel(n_jobs=-1) as parallel:
        features = parallel(delayed(generate_features)(protein_files[i], ligand_files[i]) for i in tqdm(range(len(keys))))
    # features = {}
    # for i in tqdm(range(len(keys))):
    #     features[keys[i]] = generate_features(protein_files[i], ligand_files[i])
    features_df = pd.DataFrame(features, index_col=0)
    # features = {keys[i]: result for i, result in enumerate(map(generate_features, protein_files, ligand_files))}
    print('Ready to train model')
    model = RandomForestRegressor(n_estimatators=500, n_jobs=-1)
    model.fit(features_df, pks)
    return model

def predict(model, csv_file, data_dir):
    keys, protein_files, ligand_files, pks = load_csv(csv_file, data_dir)
    features = {}
    for i in tqdm(range(len(keys))):
        features[keys[i]] = generate_features(protein_files[i], ligand_files[i])
    features_df = pd.DataFrame(features, index_col=0)
    return model.predict(features_df), pks

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv_file', help='CSV file with protein, ligand and pk data')
    parser.add_argument('--data_dir', help='Directory with protein and ligand files')
    parser.add_argument('--model_name', help='Name of the model to be saved or to be loaded')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict pK')
    args = parser.parse_args()
    if args.train:
        model = train_model(args.csv_file, args.data_dir)
        model.save(f'temp_models/{args.model_name}.pkl')
    if args.predict:
        model = RandomForestRegressor.load(f'temp_models/{args.model_name}.pkl')
        pred, true = predict(model, args.protein_file, args.ligand_file)
        print(f'Pearson: {pearsonr(pred, true)[0]}')
        print(f'Spearman: {spearmanr(pred, true)[0]}')


