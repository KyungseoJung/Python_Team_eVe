from flask import Flask, render_template

app = Flask(__name__)

# //#5 TMAP으로 지도 시각화
@app.route('/')
def index():
    return render_template('index_map_input.html')

# # //#5 현재 내 위치를 GPS로 받아와서 TMAP 지도 시각화
# @app.route('/get_location') 
# def get_location():
#     return render_template('get_location.html')

if __name__ == '__main__':
    app.run(debug=True)