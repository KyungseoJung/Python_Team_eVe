# map_visualizer.py
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
    
    # Tmap API 호출
    api_key = "5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu"    # "YOUR_API_KEY"
    url = "https://apis.skplanetx.com/tmap/routes/pedestrian?version=1&format=json&appKey=" + api_key
    payload = {
        "startX": start_point.split(',')[1],  # 경도
        "startY": start_point.split(',')[0],  # 위도
        "endX": end_point.split(',')[1],
        "endY": end_point.split(',')[0],
        "startName": "출발지",
        "endName": "도착지",
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO"
    }
    response = requests.post(url, data=payload)
    route_data = response.json()
    
    # 경로 데이터 추출
    polyline_points = []
    for feature in route_data['features']:
        geometry = feature['geometry']
        if geometry['type'] == 'LineString':
            coordinates = geometry['coordinates']
            for coordinate in coordinates:
                lat, lng = coordinate
                polyline_points.append({'lat': lat, 'lng': lng})

    return jsonify(polyline_points)

if __name__ == '__main__':
    app.run(debug=True)
