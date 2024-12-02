from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans, vq
import pandas as pd
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

load_dotenv(".env")

NUM_CLUSTERS = 10
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

class MyForm(FlaskForm):
    image = FileField("File to upload", validators=[InputRequired()])
    submit = SubmitField("Convert")

@app.route("/", methods=["GET", "POST"])
def home():
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0]), int(rgb[1]), int(rgb[2])
        )
    form = MyForm()
    if form.validate_on_submit():
        image_name = form.image.data
        image_filename = secure_filename(image_name.filename)
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image_name.save(image_path)

        image = Image.open(image_path)
        image = image.resize((150,150))
        image_array = np.array(image)
        shape = image_array.shape
        image_array = image_array.reshape(-1, shape[2]).astype(float)
        codes, _ = kmeans(image_array, NUM_CLUSTERS)
        hex_colors = [rgb_to_hex(rgb) for rgb in codes]
        vecs, _ = vq(image_array, codes)
        counts, _ = np.histogram(vecs, bins=range(len(codes) + 1))
        df = pd.DataFrame({
            'Red': codes[:, 0],
            'Green': codes[:, 1],
            'Blue': codes[:, 2],
            'Hex': hex_colors,
            'Count': counts
        })
        df = df.sort_values(by='Count', ascending=False)
        list_of_hex_colours = df["Hex"].values.tolist()
        return render_template("home.html", list=list_of_hex_colours, form=form, image_path=image_path)
    return render_template("home.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)