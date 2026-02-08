FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl gnupg2 unixodbc-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor \
    | tee /usr/share/keyrings/microsoft.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements first (better build caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your Flask application code
COPY . .

ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python init_db.py && flask run --host=0.0.0.0 --port=8080"]

