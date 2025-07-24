FROM python:3.10-slim

# Installer les dépendances système nécessaires à OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Définir les variables d'environnement utilisées par app.py
ENV API_HOST=localhost
ENV API_PORT=8000

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
