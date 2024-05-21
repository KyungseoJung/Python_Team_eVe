from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def administrator_page():
    return render_template('screen_Administrator.html')


if __name__ == '__main__':
    app.run(debug=True)
