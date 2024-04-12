from flask import Flask, request, render_template
# from flask_ngrok import run_with_ngrok
from tmap_utils import draw_polyline_on_map

app = Flask(__name__)
# run_with_ngrok(app)  # Start ngrok when app is run

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/visualize", methods=["POST"])
def visualize():
    start_point = request.form["start_point"]
    end_point = request.form["end_point"]
    draw_polyline_on_map(start_point, end_point)
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
