from flask import Flask, render_template

app = Flask(__name__)

# //#5 TMAP으로 지도 시각화
@app.route('/')
def index():
    return render_template('index_map_gps_storage.html')

if __name__ == '__main__':
    app.run(debug=True)