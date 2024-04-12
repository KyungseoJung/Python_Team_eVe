from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_route', methods=['POST'])
def get_route():
    start_point = request.form['start_point']
    end_point = request.form['end_point']
    
    # T-map API 호출하여 Polyline 경로 가져오기
    polyline = get_tmap_polyline(start_point, end_point)
    
    # Polyline을 리스트로 변환
    coordinates = polyline_to_coordinates(polyline)
    
    return jsonify({"polyline": polyline, "coordinates": coordinates})

def get_tmap_polyline(start_point, end_point):
    # T-map API 호출 및 Polyline 가져오기
    # 여기에 실제로 T-map API를 호출하는 코드를 작성해야 합니다.
    # 이 예제에서는 임의의 Polyline을 반환하도록 하겠습니다.
    return [(37.5665, 126.9780), (37.5545, 126.9707), (37.5663, 127.0047)]

def polyline_to_coordinates(polyline):
    # Polyline 좌표를 리스트로 변환
    coordinates = []
    for point in polyline:
        coordinates.append({"latitude": point[0], "longitude": point[1]})
    return coordinates

if __name__ == '__main__':
    app.run(debug=True)
