CREATE DATABASE IF NOT EXISTS emotion_db;

USE emotion_db;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    prediction VARCHAR(8) NOT NULL,
    duration FLOAT,
    timestamp DATETIME
);

CREATE TABLE IF NOT EXISTS emotions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    emotion VARCHAR(8) NOT NULL,
    timestamp DATETIME
);

CREATE TABLE IF NOT EXISTS versions (
    version VARCHAR(4) PRIMARY KEY,
    maxlen INT NOT NULL,
    duration FLOAT,
    acc_train VARCHAR(8),
    acc_val VARCHAR(8),
    timestamp DATETIME
);