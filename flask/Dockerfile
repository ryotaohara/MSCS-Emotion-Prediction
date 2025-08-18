# Use a minimal Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only dependencies first to improve caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY templates/. ./templates/
COPY app.py .

# Expose Flask default port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]
