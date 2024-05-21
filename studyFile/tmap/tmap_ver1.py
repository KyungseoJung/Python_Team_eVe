# tmap.py

import requests
import json
from flask import Flask, request, render_template

app = Flask(__name__)

# T-map API 호출을 위한 키
API_KEY = '5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu'    # 'YOUR_TMAP_API_KEY'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/draw_route', methods=['POST'])
def draw_route():
    # POST 요청에서 출발지와 도착지 좌표를 가져옴
    start_point = request.form['start']
    end_point = request.form['end']

    # 좌표값의 형식을 확인하고 수정
    start_point = start_point.replace(',', ' ')
    end_point = end_point.replace(',', ' ')
    
    # T-map API 호출
    url = f'https://apis.openapi.sk.com/tmap/routes?version=1&format=json&appKey={API_KEY}&startX={start_point}&startY={start_point}&endX={end_point}&endY={end_point}&reqCoordType=WGS84GEO&resCoordType=WGS84GEO'
    response = requests.get(url)
    data = response.json()

    print(data)  # API 응답 확인용
    try:
        if 'features' in data:  # features 키가 있는지 확인
            # Polyline 좌표값 추출
            polyline = data['features'][0]['geometry']['coordinates']
            route_coordinates = [(point[1], point[0]) for point in polyline]
            return json.dumps(route_coordinates)
        else:
            return '경로를 찾을 수 없습니다.'
    except KeyError:
        return '경로를 찾을 수 없습니다.'

if __name__ == '__main__':
    app.run(debug=True)
