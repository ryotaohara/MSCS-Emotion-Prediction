import os
from flask import Flask, request, render_template
from waitress import serve
from pymongo import MongoClient
import mysql.connector
import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for web apps
import matplotlib.pyplot as plt
import io
import base64

from datetime import datetime, timezone, timedelta
jst = timezone(timedelta(hours = 9))    # Timestamp in JST

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)           # INFO-level log is visualised

app = Flask(__name__)

# Label
label_emotions = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# API Endpoints per each environment
env = os.getenv("ENV", "local") # Default to "local" if ENV is not set
if   env == "prod":
    api_keras_mongodb = "xxxx"  # ECS service name & namespace
    api_keras_mysql   = "xxxx"  # ECS service name & namespace
    api_mongodb       = "xxxx"  # ECS service name & namespace
    api_mysql         = "xxxx"  # ECS service name & namespace
elif env == "test":
    api_keras_mongodb = "xxxx"  # ECS service name & namespace
    api_keras_mysql   = "xxxx"  # ECS service name & namespace
    api_mongodb       = "xxxx"  # ECS service name & namespace
    api_mysql         = "xxxx"  # ECS service name & namespace
else:
    api_keras_mongodb = "xxxx"  # Local container name
    api_keras_mysql   = "xxxx"  # Local container name
    api_mongodb       = "xxxx"  # Local container name
    api_mysql         = "xxxx"  # Local container name
logger.info(f"API endpoints configured for '{env}'")

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

# Landing Page
@app.route('/')
def landing():
    logger.info("=====Landing page=====")
    return render_template("landing.html")

# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    logger.info("=====Predict page=====")
    if request.method == 'POST':
        user_input = request.form['text']

        mongodb_pred = mysql_pred = "Error"
        mongodb_probs = mysql_probs = []
        mongodb_duration = mysql_duration = 0.0

        try:
            mongodb_resp     = requests.post(f"http://{api_keras_mongodb}:8888/predict",
                                             json = {"text": user_input})
            mongodb_result   = mongodb_resp.json()
            mongodb_pred     = mongodb_result.get("emotion", "Unknown")
            mongodb_probs    = mongodb_result.get("probabilities", [])
            mongodb_duration = mongodb_result.get("duration", 0)
            logger.info("Prediction obtained from MongoDB")
        except Exception as e:
            logger.error("Prediction not obtained from MongoDB")
            logger.error(f"MongoDB prediction error: {e}")

        try:
            mysql_resp     = requests.post(f"http://{api_keras_mysql}:8889/predict",
                                           json = {"text": user_input})
            mysql_result   = mysql_resp.json()
            mysql_pred     = mysql_result.get("emotion", "Unknown")
            mysql_probs    = mysql_result.get("probabilities", [])
            mysql_duration = mysql_result.get("duration", 0)
            logger.info("Prediction obtained from MySQL")
        except Exception as e:
            logger.error("Prediction not obtained from MySQL")
            logger.error(f"MySQL prediction error: {e}")

        # Plot MongoDB probabilities
        logger.info("Creating plot with MongoDB predictions")
        fig1, ax1 = plt.subplots()
        ax1.bar(label_emotions.values(), mongodb_probs, label='MongoDB')
        ax1.set_title("MongoDB Probabilities")
        plt.xticks(rotation=45)
        buf1 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf1, format='png')
        plt.close(fig1)
        buf1.seek(0)
        mongodb_plot = base64.b64encode(buf1.getvalue()).decode('utf-8')
        logger.info("Created plot with MongoDB predictions")

        # Plot MySQL probabilities
        logger.info("Creating plot with MySQL predictions")
        fig2, ax2 = plt.subplots()
        ax2.bar(label_emotions.values(), mysql_probs, label='MySQL')
        ax2.set_title("MySQL Probabilities")
        plt.xticks(rotation=45)
        buf2 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png')
        plt.close(fig2)
        buf2.seek(0)
        mysql_plot = base64.b64encode(buf2.getvalue()).decode('utf-8')
        logger.info("Created plot with MySQL predictions")

        logger.info("Returning all the results")
        return render_template(
            "predict.html",
            text             = user_input,
            mongodb_pred     = mongodb_pred,
            mongodb_plot     = mongodb_plot,
            mongodb_probs    = mongodb_probs,
            mongodb_duration = mongodb_duration,
            mysql_pred       = mysql_pred,
            mysql_plot       = mysql_plot,
            mysql_probs      = mysql_probs,
            mysql_duration   = mysql_duration
        )

    return render_template("predict.html")

# Submitting actual emotion
@app.route("/submit_actual", methods=["GET", "POST"])
def submit_actual():
    logger.info("=====Submit-Actual page=====")
    if request.method == "GET":
        return render_template(
            "submit_actual.html",
            text         = request.args.get("text", ""),
            mongodb_pred = request.args.get("mongodb_pred", ""),
            mysql_pred   = request.args.get("mysql_pred", "")
        )

    text         = request.form["text"]
    mongodb_pred = request.form["mongodb_pred"]
    mysql_pred   = request.form["mysql_pred"]
    actual       = request.form["actual"]
    now_jst      = datetime.now(jst)

    # MongoDB
    mongodb_conn = get_mongodb_conn()
    mongodb_db   = mongodb_conn["emotion_db"]
    col_emotions = mongodb_db["emotions"]
    col_emotions.insert_one({
        "text"     : text,
        "emotion"  : actual,
        "timestamp": now_jst + timedelta(hours = 9)
    })
    logger.info("Actual emotion inserted into MongoDB.")
    mongodb_conn.close()
    logger.info("Connection with MongoDB closed.")

    # MySQL
    mysql_conn = get_mysql_conn()
    mysql_cur  = mysql_conn.cursor()
    mysql_cur.execute(
        "INSERT INTO emotions (text, emotion, timestamp) VALUES (%s, %s, %s)",
        (text, actual, now_jst)
    )
    logger.info("Actual emotion inserted into MySQL.")
    mysql_conn.commit()
    mysql_cur.close()
    mysql_conn.close()
    logger.info("Connection with MySQL closed.")

    logger.info("Actual emotion recorded")
    return render_template('thank_you.html', record={
        "text"        : text,
        "mongodb_pred": mongodb_pred,
        "mysql_pred"  : mysql_pred,
        "actual"      : actual
    })

# Retraing model
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    logger.info("=====Retrain page=====")
    if request.method == "POST":
        # MongoDB
        logger.info("Model retraining started for MongoDB")
        mongodb_resp      = requests.post(f"http://{api_keras_mongodb}:8888/retrain")
        mongodb_version   = mongodb_resp.json().get("version")
        mongodb_duration  = mongodb_resp.json().get("duration")
        mongodb_acc_train = mongodb_resp.json().get("acc_train")
        mongodb_acc_val   = mongodb_resp.json().get("acc_val")
        logger.info("Model retraining finished for MongoDB")

        # MySQL
        logger.info("Model retraining started for MySQL")
        mysql_resp      = requests.post(f"http://{api_keras_mysql}:8889/retrain")
        mysql_version   = mysql_resp.json().get("version")
        mysql_duration  = mysql_resp.json().get("duration")
        mysql_acc_train = mysql_resp.json().get("acc_train")
        mysql_acc_val   = mysql_resp.json().get("acc_val")
        logger.info("Model retraining finished for MySQL")

        logger.info("Model retraining finished for both MongoDB and MySQL")
        return render_template("retrain.html",
                               mongodb_version   = mongodb_version,
                               mongodb_duration  = mongodb_duration,
                               mongodb_acc_train = mongodb_acc_train,
                               mongodb_acc_val   = mongodb_acc_val,
                               mysql_version   = mysql_version,
                               mysql_duration  = mysql_duration,
                               mysql_acc_train = mysql_acc_train,
                               mysql_acc_val   = mysql_acc_val)
    return render_template("retrain.html")

# Prediction Record Page in MongoDB
@app.route("/records_predictions_mongodb")
def records_predictions_mongodb():
    logger.info("=====Prediction records page for MongoDB=====")
    start_time = datetime.now(jst)
    page     = int(request.args.get("page", 1))
    query    = request.args.get("query", "").strip()
    per_page = 100
    skip     = (page - 1) * per_page

    filter_cond = {"text": {"$regex": query, "$options": "i"}} if query else {}

    # count total docs to compute total_pages
    mongodb_conn = get_mongodb_conn()
    mongodb_db = mongodb_conn["emotion_db"]
    col_predictions = mongodb_db["predictions"]
    total_records = col_predictions.count_documents(filter_cond)
    total_pages   = (total_records + per_page - 1) // per_page

    records = list(
        col_predictions
          .find(filter_cond, {"_id": 0})
          .skip(skip)
          .limit(per_page)
    )
    mongodb_conn.close()

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)

    return render_template(
        "records_predictions.html",
        records     = records,
        page        = page,
        total_pages = total_pages,
        query       = query,
        filter_time = duration,
        source_name = "MongoDB"
    )

# Prediction Record Page in MySQL
@app.route("/records_predictions_mysql")
def records_predictions_mysql():
    logger.info("=====Prediction records page for MySQL=====")
    start_time = datetime.now(jst)
    page     = int(request.args.get("page", 1))
    query    = request.args.get("query", "").strip()
    per_page = 100
    offset   = (page - 1) * per_page

    mysql_conn = get_mysql_conn()
    mysql_cur  = mysql_conn.cursor(dictionary=True)

    like_expr = f"%{query}%"
    where_clause = "WHERE text LIKE %s" if query else ""
    params = [like_expr] if query else []

    # Count total records
    mysql_cur.execute(f"SELECT COUNT(*) as count FROM predictions {where_clause}", params)
    total_records = mysql_cur.fetchone()["count"]
    total_pages = (total_records + per_page - 1) // per_page

    # Fetch paginated slice
    mysql_cur.execute(f"""
        SELECT text, prediction, duration, timestamp
        FROM predictions
        {where_clause}
        ORDER BY id
        LIMIT %s OFFSET %s
        """,
        params + [per_page, offset]
    )
    records = mysql_cur.fetchall()

    mysql_cur.close()
    mysql_conn.close()

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)

    return render_template(
        "records_predictions.html",
        records     = records,
        page        = page,
        total_pages = total_pages,
        query       = query,
        filter_time = duration,
        source_name = "MySQL"
    )

# Emotion Record Page in MongoDB
@app.route("/records_emotions_mongodb")
def records_emotions_mongodb():
    logger.info("=====Emotion records page for MongoDB=====")
    start_time = datetime.now(jst)
    # grab page number from query, default to 1
    page     = int(request.args.get('page', 1))
    query    = request.args.get('query', '').strip()
    per_page = 100
    skip     = (page - 1) * per_page

    filter_cond = {"text": {"$regex": query, "$options": "i"}} if query else {}

    # count total docs to compute total_pages
    mongodb_conn = get_mongodb_conn()
    mongodb_db   = mongodb_conn["emotion_db"]
    col_emotions = mongodb_db["emotions"]
    total_records = col_emotions.count_documents(filter_cond)
    total_pages   = (total_records + per_page - 1) // per_page

    # fetch only this page’s slice
    records = list(
        col_emotions
          .find(filter_cond, {"_id": 0})
          .skip(skip)
          .limit(per_page)
    )
    mongodb_conn.close()

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)

    return render_template(
        "records_emotions.html",
        records     = records,
        page        = page,
        total_pages = total_pages,
        query       = query,
        filter_time = duration,
        source_name = "MongoDB"
    )

# Emotion Record Page in MySQL
@app.route("/records_emotions_mysql")
def records_emotions_mysql():
    logger.info("=====Emotion records page for MySQL=====")
    start_time = datetime.now(jst)
    page     = int(request.args.get("page", 1))
    query    = request.args.get("query", "").strip()
    per_page = 100
    offset   = (page - 1) * per_page

    mysql_conn = get_mysql_conn()
    mysql_cur  = mysql_conn.cursor(dictionary=True)

    like_expr = f"%{query}%"
    where_clause = "WHERE text LIKE %s" if query else ""
    params = [like_expr] if query else []

    # Count total records
    mysql_cur.execute(f"SELECT COUNT(*) as count FROM emotions {where_clause}", params)
    total_records = mysql_cur.fetchone()["count"]
    total_pages = (total_records + per_page - 1) // per_page

    # Fetch paginated slice
    mysql_cur.execute(f"""
        SELECT text, emotion, timestamp
        FROM emotions
        {where_clause}
        ORDER BY id
        LIMIT %s OFFSET %s
        """,
        params + [per_page, offset]
    )
    records = mysql_cur.fetchall()
    mysql_cur.close()
    mysql_conn.close()

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)

    return render_template(
        "records_emotions.html",
        records     = records,
        page        = page,
        total_pages = total_pages,
        query       = query,
        filter_time = duration,
        source_name = "MySQL"
    )

# Emotion Record Page in MongoDB
@app.route("/records_versions_mongodb")
def records_versions_mongodb():
    logger.info("=====Version record page for MongoDB=====")
    start_time = datetime.now(jst)
    # grab page number from query, default to 1
    page     = int(request.args.get("page", 1))
    query    = request.args.get("query", "").strip()
    per_page = 100
    skip     = (page - 1) * per_page

    filter_cond = {"text": {"$regex": query, "$options": "i"}} if query else {}

    # count total docs to compute total_pages
    mongodb_conn = get_mongodb_conn()
    mongodb_db   = mongodb_conn["emotion_db"]
    col_versions = mongodb_db["versions"]
    total_records = col_versions.count_documents(filter_cond)
    total_pages   = (total_records + per_page - 1) // per_page

    # fetch only this page’s slice
    records = list(
        col_versions
          .find(filter_cond, {"_id": 0})
          .skip(skip)
          .limit(per_page)
    )
    mongodb_conn.close()

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)

    return render_template(
        "records_versions.html",
        records     = records,
        page        = page,
        total_pages = total_pages,
        query       = query,
        filter_time = duration,
        source_name = "MongoDB"
    )

# Emotion Record Page in MySQL
@app.route("/records_versions_mysql")
def records_versions_mysql():
    logger.info("=====Version record page for MySQL=====")
    start_time = datetime.now(jst)
    page     = int(request.args.get('page', 1))
    query    = request.args.get('query', '').strip()
    per_page = 100
    offset   = (page - 1) * per_page

    mysql_conn = get_mysql_conn()
    mysql_cur  = mysql_conn.cursor(dictionary=True)

    like_expr = f"%{query}%"
    where_clause = "WHERE text LIKE %s" if query else ""
    params = [like_expr] if query else []

    # Count total records
    mysql_cur.execute(f"SELECT COUNT(*) as count FROM versions {where_clause}", params)
    total_records = mysql_cur.fetchone()["count"]
    total_pages = (total_records + per_page - 1) // per_page

    # Fetch paginated slice
    mysql_cur.execute(f"""
        SELECT version, maxlen, duration, acc_train, acc_val, timestamp
        FROM versions
        {where_clause}
        ORDER BY version
        LIMIT %s OFFSET %s
        """,
        params + [per_page, offset]
    )
    records = mysql_cur.fetchall()
    mysql_cur.close()
    mysql_conn.close()

    end_time = datetime.now(jst)
    duration = round((end_time - start_time).total_seconds(), 3)

    return render_template(
        "records_versions.html",
        records     = records,
        page        = page,
        total_pages = total_pages,
        query       = query,
        filter_time = duration,
        source_name = "MySQL"
    )


if __name__ == "__main__":
    serve(app, host = "0.0.0.0", port = 8080)
