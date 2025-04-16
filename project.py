from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import os

# Flask instance
app = Flask(__name__)

######### VALEUR MANQUANTES #########
def traiter_valeurs_manquantes(df):
    msno.matrix(df)  # afficher les valeurs manquantes
    plt.show()
    print(df.isnull().mean() * 100)  # obtenir le pourcentage des valeurs manquantes
    
    # Traitement automatique des valeurs manquantes
    for colonne in df.columns:
        if df[colonne].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[colonne]):
                moyenne = df[colonne].mean()
                df[colonne].fillna(moyenne, inplace=True)
            else:
                mode = df[colonne].mode()[0]
                df[colonne].fillna(mode, inplace=True)
    return df

########### VALEUR ABERANTES ############
def traiter_valeurs_aberantes(df):
    sns.boxplot(x=df['Age'])
    plt.title("Boxplot des valeurs aberrantes")
    plt.show()
    
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    
    outliers = df[(df['Age'] < borne_inf) | (df['Age'] > borne_sup)]
    print(f"Nombre de valeurs aberrantes dans 'Age' : {len(outliers)}")
    
    df['Age'] = df['Age'].apply(lambda x: df['Age'].median() if x < borne_inf or x > borne_sup else x)
    return df  # Retourner le DataFrame complet

########### VALEUR DUPLIQUEES ############
def traiter_valeurs_dupliqees(df):
    print(df.duplicated().any())  # vérifie la présence de doublons
    print(df.duplicated().sum())  # nombre de doublons
    print(df[df.duplicated()])  # afficher les doublons
    df = df.drop_duplicates(keep='last')  # Garder la dernière occurrence
    return df   

########### NORMALISATION DES DONNEES ############
def normaliser(df):
    print(df.head())
    scaler = MinMaxScaler()
    
    colonnes = [col for col in ['PassengerId', 'Pclass'] if col in df.columns]
    if colonnes:
        df[colonnes] = scaler.fit_transform(df[colonnes])
    
    print(df.head())
    df.to_csv('data_normalized.csv', index=False)
    return df

# Pipeline de traitement des données
def pipeline(df):
    df = traiter_valeurs_manquantes(df)
    df = traiter_valeurs_aberantes(df)
    df = traiter_valeurs_dupliqees(df)
    df = normaliser(df)
    return df

######### ROUTE FLASK ##########
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/treatement/')
def treatement():
    return render_template('treatement.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier reçu'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fichier invalide'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Le fichier doit être au format CSV'}), 400

    try:
        content = file.read()
        if not content:
            return jsonify({'error': 'Le fichier est vide'}), 400

        stream = StringIO(content.decode('UTF8'), newline=None)
        df = pd.read_csv(stream)

        if df.empty:
            return jsonify({'error': 'Le fichier CSV ne contient pas de données'}), 400
        
        df = pipeline(df)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return jsonify({'error': 'Le fichier CSV doit contenir au moins une colonne numérique'}), 400

        selected_column = numeric_columns[0]
        df[f'{selected_column}_double'] = df[selected_column] * 2

        df = df.replace({np.nan: None})
        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
          return jsonify({
            'message': 'Traitement terminé avec succès',
            'data': df.to_dict(orient='records'),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates_removed': df.duplicated().sum()
        })

    df.to_csv('donnees_traitees.csv', index=False)  

if __name__ == '__main__':
    port= int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)
