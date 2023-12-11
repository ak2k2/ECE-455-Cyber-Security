import io
import os
import pickle

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

from embedding import get_embedding_from_face
from helpers import decrypt_message, encrypt_message, get_config
from view_schema import validate_two_factor_login

CONFIG = get_config()
FERNET_KEY = CONFIG["encryption"]["key"].encode()


# Initialize Flask and SQLAlchemy
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(basedir, "user")
db_uri = "sqlite:///" + os.path.join(db_dir, "users.db")
app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
app.config["SECRET_KEY"] = "your_secret_key"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    image_1 = db.Column(db.LargeBinary)
    image_2 = db.Column(db.LargeBinary)
    image_3 = db.Column(db.LargeBinary)
    embedding = db.Column(db.LargeBinary)  # New field for embedding

    def encrypt_and_store_embedding(self, embedding):
        global FERNET_KEY

        if embedding is not None:
            # Serialize the numpy array and then encrypt it
            pickled_embedding = pickle.dumps(embedding)
            self.embedding = encrypt_message(pickled_embedding, FERNET_KEY)

    def get_decrypted_embedding(self):
        global FERNET_KEY
        if self.embedding:
            decrypted_embedding = decrypt_message(self.embedding, FERNET_KEY)
            return pickle.loads(decrypted_embedding)
        else:
            return None

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def encrypt_and_store_images(self, image1, image2, image3):
        global FERNET_KEY

        if image1 is not None:
            self.image_1 = encrypt_message(image1, FERNET_KEY)
        if image2 is not None:
            self.image_2 = encrypt_message(image2, FERNET_KEY)
        if image3 is not None:
            self.image_3 = encrypt_message(image3, FERNET_KEY)

    def get_decrypted_images(self):
        global FERNET_KEY
        return {
            "image_1": decrypt_message(self.image_1, FERNET_KEY)
            if self.image_1
            else None,
            "image_2": decrypt_message(self.image_2, FERNET_KEY)
            if self.image_2
            else None,
            "image_3": decrypt_message(self.image_3, FERNET_KEY)
            if self.image_3
            else None,
        }


# Login manager
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Home route
@app.route("/")
def index():
    return render_template("index.html")


# Registration route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists.")
            return redirect(url_for("register"))

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful.")
        return redirect(url_for("login"))

    return render_template("register.html")


# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("capture"))

        flash("Invalid username or password.")

    return render_template("login.html")


@app.route("/capture", methods=["GET", "POST"])
@login_required
def capture():
    if request.method == "POST":
        image_num = request.form.get("image_num")
        image_file = request.files["image"] if "image" in request.files else None

        if image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Assuming you want to capture 3 images before proceeding to 2FA
            max_images = 3

            if image_num == "1":
                embedding = get_embedding_from_face(image)
                current_user.encrypt_and_store_images(image_bytes, None, None)
                current_user.encrypt_and_store_embedding(embedding)
            elif image_num == "2":
                current_user.encrypt_and_store_images(None, image_bytes, None)
            elif image_num == "3":
                current_user.encrypt_and_store_images(None, None, image_bytes)

            db.session.commit()

            # Return a success status after processing each image
            return jsonify({"status": "success", "image_num": image_num})
        else:
            return jsonify({"status": "error", "message": "No image provided"})

    return render_template("capture.html", username=current_user.username)


@app.route("/2fa", methods=["GET", "POST"])
@login_required
def process_2fa_login():
    if request.method == "POST":
        # Receive the image from the form
        image_file = request.files["image"] if "image" in request.files else None

        if image_file:
            # Convert the image to PIL format and validate
            attempted_face = Image.open(io.BytesIO(image_file.read())).convert("RGB")
            username = (
                current_user.username
            )  # Get the current logged-in user's username
            result = validate_two_factor_login(username, attempted_face)

            if result:
                return jsonify({"status": "success", "redirect": url_for("dashboard")})
            else:
                return jsonify(
                    {"status": "failure", "message": "2FA Verification Failed"}
                )

        else:
            return jsonify({"status": "failure", "message": "No image provided"})

    return render_template("2fa.html")


# Protected dashboard route
@app.route("/dashboard")
@login_required
def dashboard():
    return f"Welcome to your dashboard, {current_user.username}!"


# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# Create the SQLite database
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run()
