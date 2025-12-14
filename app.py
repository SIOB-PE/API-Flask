from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configurações
# Usando o seu MongoDB Atlas (o mesmo do generate_data.py)
# MONGO_URI = "mongodb+srv://vyniciushenrique:siobpe@siob-pe.ffyikjl.mongodb.net/?appName=SIOB-PE"
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "siob_pe_db"
COLLECTION_NAME = "Ocorrencia"
MODEL_FILE = "model.pkl"

modelo = None
label_encoder = None

def carregar_modelo():
    global modelo, label_encoder
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                data = pickle.load(f)
                modelo = data["pipeline"]
                label_encoder = data["label_encoder"]
            print("Modelo carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
    else:
        print("Aviso: 'model.pkl' não encontrado. Execute train_model.py.")

carregar_modelo()

# --- NOVO ENDPOINT PARA OS GRÁFICOS ---
@app.route('/api/dados/dashboard', methods=['GET'])
def dados_dashboard():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        colecao = db[COLLECTION_NAME]

        # 1. Gráfico de Rosca: Contagem por Tipo
        pipeline_natureza = [
            {"$group": {"_id": "$tipoNaturezaOcorrencia", "count": {"$sum": 1}}}
        ]
        natureza_data = list(colecao.aggregate(pipeline_natureza))
        natureza_dict = {str(item["_id"]): item["count"] for item in natureza_data if item["_id"]}

        # 2. Gráfico de Barras: Top 5 Bairros
        pipeline_bairro = [
            {"$group": {"_id": "$bairro", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        bairro_data = list(colecao.aggregate(pipeline_bairro))
        bairro_dict = {str(item["_id"]): item["count"] for item in bairro_data if item["_id"]}

        return jsonify({
            "natureza": natureza_dict,
            "bairros": bairro_dict
        }), 200

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/predizer', methods=['POST'])
def predizer():
    if not modelo:
        return jsonify({"erro": "Modelo indisponível"}), 503

    data = request.get_json()
    
    try:
        df_input = pd.DataFrame([data])
        y_encoded = modelo.predict(df_input)[0]
        y_prob = modelo.predict_proba(df_input)[0]
        resultado_classe = label_encoder.inverse_transform([y_encoded])[0]
        
        probs = {str(c): float(p) for c, p in zip(label_encoder.classes_, y_prob)}

        return jsonify({
            "previsao": resultado_classe,
            "confianca": probs
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/modelo/importancia', methods=['GET'])
def feature_importance():
    if not modelo:
        return jsonify({}), 200
    try:
        classifier = modelo.named_steps['classifier']
        preprocessor = modelo.named_steps['preprocessor']
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        all_names = list(cat_names) + ["hora"]
        importances = classifier.feature_importances_
        
        result = [
            {"feature": name, "score": float(score)} 
            for name, score in zip(all_names, importances) if float(score) > 0.001
        ]
        return jsonify(sorted(result, key=lambda x: x["score"], reverse=True)[:10])
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)