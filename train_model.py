import pandas as pd
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
import os

# CONFIGURAÇÃO
# Certifique-se de que é a mesma URI usada no application.yml do Spring
MONGO_URI = "mongodb+srv://vyniciushenrique:siobpe@siob-pe.ffyikjl.mongodb.net/?appName=SIOB-PE"
DB_NAME = "siob_pe_db" # Nome do banco (verifique no seu Mongo)
COLLECTION_NAME = "Ocorrencia"

def treinar():
    print("Conectando ao MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    colecao = db[COLLECTION_NAME]

    # Buscar apenas os campos necessários para economizar memória
    cursor = colecao.find({}, {
        "bairro": 1, 
        "municipio": 1, 
        "dataHoraAcionamento": 1, 
        "tipoNaturezaOcorrencia": 1, 
        "_id": 0
    })
    
    dados = list(cursor)
    
    if not dados:
        print("ERRO: Nenhuma ocorrência encontrada no banco para treinar.")
        return

    print(f"Processando {len(dados)} registros...")

    lista = []
    for d in dados:
        # Tratamento de Data/Hora (Extrair a Hora)
        hora = 12 # Default
        dt_raw = d.get("dataHoraAcionamento")
        
        if dt_raw:
            try:
                # Se vier como string (ISO) do Java
                if isinstance(dt_raw, str):
                    dt = datetime.fromisoformat(dt_raw.replace("Z", ""))
                    hora = dt.hour
                # Se vier como objeto datetime nativo do Mongo
                elif isinstance(dt_raw, datetime):
                    hora = dt_raw.hour
            except Exception as e:
                pass

        lista.append({
            "bairro": d.get("bairro", "NaoInformado"),
            "municipio": d.get("municipio", "NaoInformado"),
            "hora": hora,
            "target": d.get("tipoNaturezaOcorrencia", "OUTROS")
        })

    df = pd.DataFrame(lista)

    # Separação X (Features) e y (Target)
    X = df[["bairro", "municipio", "hora"]]
    y = df["target"]

    # Encode do Target (Texto -> Número)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Pipeline de Processamento
    categorical_features = ["bairro", "municipio"]
    numeric_features = ["hora"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric='mlogloss'))
    ])

    print("Treinando modelo XGBoost...")
    pipeline.fit(X, y_encoded)

    # Salvar
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "label_encoder": label_encoder
        }, f)

    print("Sucesso! Modelo salvo em 'model.pkl'.")
    print("Classes detectadas:", list(label_encoder.classes_))

if __name__ == "__main__":
    treinar()