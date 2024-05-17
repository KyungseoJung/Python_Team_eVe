from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('screen2_Driver.html')

if __name__ == '__main__':
    app.run(debug=True)
