
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_login import LoginManager
from itsdangerous import URLSafeTimedSerializer


db = SQLAlchemy()
mail = Mail()
login_manager = LoginManager()

s = None

def initialize_serializer(app):
    """Initializes the serializer with the app's secret key."""
    global s
    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

