import os
import json
import string
import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Dense, Embedding, Dropout, Bidirectional, BatchNormalization

from flask import Flask, request, jsonify
from waitress import serve

from datetime import datetime, timezone, timedelta
jst = timezone(timedelta(hours = 9))    # Timestamp in JST

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)           # INFO-level log is visualised

app = Flask(__name__)

# Prep for NLP
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
num_words = 50000

# Label-emotion mapping
label_emotions = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
emotion_labels = {
    'sadness' : 0,
    'joy'     : 1,
    'love'    : 2,
    'anger'   : 3,
    'fear'    : 4,
    'surprise': 5
}

# Environment variables
epochs     = os.getenv("epochs", 3)
batch_size = os.getenv("batch_size", 1024)

# API Endpoints for each environment
env = os.getenv("ENV", "local") # Default to 'local' if ENV is not set
env = env.lower()
if   env == "prod":
    api_mongodb       = "localhost" # 'localhost' due to being in the same ECS task/service
    api_mysql         = "localhost" # 'localhost' due to being in the same ECS task/service
elif env == "test":
    api_mongodb       = "localhost" # 'localhost' due to being in the same ECS task/service
    api_mysql         = "localhost" # 'localhost' due to being in the same ECS task/service
else:
    api_mongodb       = "xxxx"      # Local container name
    api_mysql         = "xxxx"      # Local container name
logger.info(f"API endpoints configured for '{env}'")

# Initial database setup
db = os.getenv("DB", "MySQL")
db = db.lower()
now_jst = datetime.now(jst)
if db == "mongodb":
    logger.info("MongoDB selected as the database")
    from pymongo import MongoClient

    # MongoDB connection function
    def get_mongodb_conn():
        logger.info(f"Connecting to the MongoDB host '{api_mongodb}' on the'{env}' environment")
        client = MongoClient(
                    host       = api_mongodb,
                    port       = 27017,
                    username   = "xxxx",
                    password   = "xxxx",
                    authSource = "xxxx"
                )
        logger.info(f"Connected to the MongoDB host '{api_mongodb}' on the '{env}' environment")
        return client

    # MongoDB connection
    mongodb_conn    = get_mongodb_conn()
    mongodb_db      = mongodb_conn["emotion_db"]
    col_predictions = mongodb_db["predictions"]
    col_emotions    = mongodb_db["emotions"]
    col_versions    = mongodb_db["versions"]

    # Insert initial emotion dataset if empty
    if col_emotions.count_documents({}) == 0:
        emotions_df = pd.read_csv("emotions.csv")
        emotions_df['emotion'] = emotions_df['label'].map(label_emotions)
        emotions_df['timestamp'] = None
        emotions_dict = emotions_df.to_dict(orient='records')
        col_emotions.insert_many(emotions_dict)

    # Insert initial version if empty
    if col_versions.count_documents({}) == 0:
        version_dict = {"version"   : "v001",
                        "maxlen"    : 79,
                        "duration"  : None,
                        "acc_train" : "93.4%",
                        "acc_val"   : "91.2%",
                        "timestamp" : now_jst + timedelta(hours = 9)}
        col_versions.insert_one(version_dict)
        
    mongodb_conn.close()
    logger.info("MongoDB setup complete")

else:
    logger.info("MySQL selected as the database")
    import mysql.connector

    # MySQL connection function
    def get_mysql_conn():
        logger.info(f"Connecting to the MySQL host '{api_mysql}' on the '{env}' environment")
        mysql_conn = mysql.connector.connect(
                        host     = api_mysql,
                        port     = 3306,
                        user     = "xxxx",
                        password = "xxxx",
                        database = "emotion_db"
                    )
        logger.info(f"Connected to the MySQL host '{api_mysql}' on the '{env}' environment")
        return mysql_conn

    # MySQL connection
    mysql_conn = get_mysql_conn()
    mysql_cur = mysql_conn.cursor()

    # Insert intial emotion dataset into MySQL if empty
    mysql_cur.execute("SELECT COUNT(*) FROM emotions")
    if mysql_cur.fetchone()[0] == 0:
        emotions_df = pd.read_csv("emotions.csv")
        emotions_df['emotion'] = emotions_df['label'].map(label_emotions)
        emotions_df['timestamp'] = None
        for _, row in emotions_df.iterrows():
            mysql_cur.execute(
                "INSERT INTO emotions (text, emotion, timestamp) VALUES (%s, %s, %s)",
                (row["text"], row["emotion"], row["timestamp"])
            )

    # Insert initial version if empty
    mysql_cur.execute("SELECT COUNT(*) FROM versions")
    if mysql_cur.fetchone()[0] == 0:
        mysql_cur.execute(
            "INSERT INTO versions (version, maxlen, acc_train, acc_val, timestamp) VALUES (%s, %s, %s, %s, %s)",
            ("v001", 79, "93.4%", "91.2%", now_jst)
        )

    mysql_conn.commit()
    mysql_cur.close()
    mysql_conn.close()
    logger.info("MySQL setup complete")


# Function to preprocess user input
def preprocess_text(text):
    t = text.lower().strip()
    t = t.translate(str.maketrans('','',string.punctuation))
    tokens = word_tokenize(t)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    logger.info("=====Predict page=====")
    start_time = datetime.now(jst)
    data = request.get_json()
    input = data.get("text")
    if not input:
        return jsonify({"error": "Missing text"}), 400

    # Parameter version
    if db == "mongodb":
        mongodb_conn = get_mongodb_conn()
        mongodb_db   = mongodb_conn["emotion_db"]
        col_versions = mongodb_db["versions"]
        param_ver    = max(col_versions.distinct("version"))
        maxlen       = col_versions.find_one({"version": param_ver})["maxlen"]
    else:
        mysql_conn = get_mysql_conn()
        mysql_cur  = mysql_conn.cursor(dictionary=True)
        mysql_cur.execute("SELECT * FROM versions ORDER BY timestamp DESC LIMIT 1")
        row        = mysql_cur.fetchone()
        param_ver  = row["version"]
        maxlen     = row["maxlen"]
    logger.info(f"Current version obtained from {db}.")

    # Paths
    model_path     = os.path.join(os.getcwd(), f"models/{param_ver}.keras")
    tokenizer_path = os.path.join(os.getcwd(), f"tokenizers/{param_ver}.json")

    # Load model and tokenizer
    model = load_model(model_path)
    with open(tokenizer_path, "r") as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
    logger.info("Model and tokenizer loaded")

    # Preprocessing
    text          = preprocess_text(input)
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded   = pad_sequences(text_sequence, maxlen=maxlen, padding = 'post')

    prediction      = model.predict([text_padded])
    predicted_index = int(prediction.argmax(axis=-1)[0])
    emotion         = label_emotions.get(predicted_index, "unknown")

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)
    now_jst = datetime.now(jst)

    # Saving to the database
    if db == "mongodb":
        col_predictions = mongodb_db["predictions"]
        col_predictions.insert_one({
            "text"       : input,
            "prediction" : emotion,
            "duration"   : duration,
            "timestamp"  : now_jst + timedelta(hours = 9)
        })
        logger.info(f"Prediction saved to {db}.")
        mongodb_conn.close()
        logger.info(f"Connection to {db} closed.")
        
    else:
        mysql_cur.execute(
            "INSERT INTO predictions (text, prediction, duration, timestamp) VALUES (%s, %s, %s, %s)",
            (input, emotion, duration, now_jst)
        )
        logger.info(f"Prediction saved to {db}.")
        mysql_conn.commit()
        mysql_cur.close()
        mysql_conn.close()
        logger.info(f"Connection to {db} closed.")

    logger.info("Prediction finished, sending the information back to the main server")
    return jsonify({
        "emotion"       : emotion,
        "probabilities" : prediction.tolist()[0],
        "duration"      : duration
    })

# Retrain endpoint
@app.route("/retrain", methods=["POST"])
def retrain():
    logger.info("=====Retrain page=====")
    start_time = datetime.now(jst)

    # Load data
    if db == "mongodb":
        mongodb_conn = get_mongodb_conn()
        mongodb_db   = mongodb_conn["emotion_db"]
        col_emotions = mongodb_db["emotions"]
        docs = list(col_emotions.find({}, {"_id": 0}))
        df = pd.DataFrame(docs)
    else:
        mysql_conn = get_mysql_conn()
        df = pd.read_sql("SELECT text, emotion FROM emotions", mysql_conn)
    logger.info(f"Emotion records obtained from {db}.")

    # Preprocessing
    df["label"] = df["emotion"].map(emotion_labels)
    df["text"] = df["text"].apply(preprocess_text)

    # Tokenization and Padding
    x = df["text"]
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    maxlen = max(len(s) for s in x)
    x = pad_sequences(x, maxlen=maxlen, padding="post")
    y = df["label"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Model Building
    model = Sequential()
    model.add(Embedding(input_dim = num_words, output_dim = 100))
    model.add(Bidirectional(GRU(128)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation = "softmax"))
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    # Model training
    logger.info("Retraining started.")
    hist = model.fit(x_train, y_train,
                     epochs = epochs, batch_size = batch_size, verbose = 0, # epochs = 1 for test
                     validation_data = (x_test, y_test))
    logger.info("Retraining finished.")
    acc_train = hist.history['accuracy'][-1]
    acc_train = f"{acc_train * 100:.1f}%"
    acc_val   = hist.history['val_accuracy'][-1]
    acc_val   = f"{acc_val * 100:.1f}%"
    
    # Current version
    if db == "mongodb":
        col_versions = mongodb_db["versions"]
        ver_current = max(col_versions.distinct("version"))
    else:
        mysql_cur = mysql_conn.cursor(dictionary=True)
        mysql_cur.execute("SELECT version FROM versions ORDER BY timestamp DESC LIMIT 1")
        ver_current = mysql_cur.fetchone()['version']
    logger.info(f"Current version: '{ver_current}'.")

    # Next version
    ver_current = ver_current.replace('v', '')
    ver_current = int(ver_current)
    ver_current = ver_current + 1
    ver_current = str(ver_current)
    ver_current = (3 - len(ver_current)) * "0" + ver_current
    ver_next = 'v' + ver_current
    logger.info(f"New version: '{ver_next}'.")

    # Paths
    model_path     = os.path.join(os.getcwd(), f"models/{ver_next}.keras")
    tokenizer_path = os.path.join(os.getcwd(), f"tokenizers/{ver_next}.json")

    # Save model and tokenizer
    model.save(model_path)
    with open(tokenizer_path, 'w') as f:
        tokenizer_json = tokenizer.to_json()
        json.dump(tokenizer_json, f)
    logger.info("New model and tokenizer saved.")

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)
    now_jst  = datetime.now(jst)

    # Record version in the database
    if db == "mongodb":
        col_versions.insert_one({
            "version"   : ver_next,
            "maxlen"    : maxlen,
            "duration"  : duration,
            "acc_train" : acc_train,
            "acc_val"   : acc_val,
            "timestamp" : now_jst + timedelta(hours = 9)
        })
        logger.info(f"New version, inserted into {db}.")
        mongodb_conn.close()
        logger.info(f"Connection with {db} closed.")
    else:
        mysql_cur.execute(
            "INSERT INTO versions (version, maxlen, duration, acc_train, acc_val, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
            (ver_next, maxlen, duration, acc_train, acc_val, now_jst)
        )
        logger.info(f"New version, inserted into {db}.")
        mysql_conn.commit()
        mysql_cur.close()
        mysql_conn.close()
        logger.info(f"Connection with {db} closed.")

    logger.info("Version updated, sending the information back to the main server.")
    return jsonify({
        "status"    : "success",
        "version"   : ver_next,
        "acc_train" : acc_train,
        "acc_val"   : acc_val,
        "duration"  : duration
    })

if __name__ == '__main__':
    logger.info(f"Starting the web server with {db} as the database.")
    if db == "mongodb":
        serve(app, host = "0.0.0.0", port = 8888)
    else:
        serve(app, host = "0.0.0.0", port = 8889)
