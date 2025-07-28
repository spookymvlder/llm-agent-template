from flask import Flask, flash, redirect, render_template, request, session, send_file, jsonify
from flask_session import Session

import os

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

Session(app)

@app.route("/", methods=["GET", "POST"])
def index():
    # Define form fields
    # Call agent
    return #render_template(".html")