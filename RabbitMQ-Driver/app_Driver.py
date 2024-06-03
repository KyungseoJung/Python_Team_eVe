'''

#35 고객 화면과 Driver 화면에서 코드 추가 - 모바일 웹 화면으로 확인할 때, 페이지 연결 및 구조가 적절히 보이도록

#41 현재시간 기준으로 배달 관련 csv 파일 3가지 중 하나를 가져오기

'''

'''
#60
cmd 창에 ipconfig를 통해 IPv4 주소를 찾아

1) 모바일로 실행 시,
screen_Driver_home.html의 url 주소는 IPv4 주소를 작성하도록
-> 매일 수정되니까 실행할 때마다 확인 - 수정 - 실행
-> app_Driver.py 파일의 port 번호와 screen_Driver_home.html 의 port번호를 동일하게 맞추기

2) RabbitMQ 주소 맞추기
var ws = new WebSocket("ws://127.0.0.1:15674/ws");  -> 로컬로 연결되어 있는 것
IPv4 주소를 찾아서 아래와 같이 수정해주기
예를 들어, 192.168.50.178이라면, var ws = new WebSocket("ws://192.168.50.178:15674/ws");

'''
from flask import Flask, render_template, send_from_directory, abort, jsonify
import pandas as pd
from datetime import datetime   #41
import json # 44

app = Flask(__name__)

#41 현재시간 기준으로 배달 관련 csv 파일 가져오기
# Define the base path for the CSV files
BASE_PATH = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"

def get_current_time_in_minutes():
    now = datetime.now()
    return now.hour * 60 + now.minute

def select_csv_file():
    current_time = get_current_time_in_minutes()
    if 0 <= current_time < 480:
        return BASE_PATH + "Truck_routes_E_01.csv"
    elif 480 <= current_time < 960:
        return BASE_PATH + "Truck_routes_E_02.csv"
    elif 960 <= current_time < 1440:
        return BASE_PATH + "Truck_routes_E_03.csv"
    else:
        raise ValueError("Invalid time range")

def select_json_file():
    current_time = get_current_time_in_minutes()
    if 0 <= current_time < 480:
        return BASE_PATH + "drawroute_E_01.json"
    elif 480 <= current_time < 960:
        return BASE_PATH + "drawroute_E_02.json"
    elif 960 <= current_time < 1440:
        return BASE_PATH + "drawroute_E_03.json"
    else:
        raise ValueError("Invalid time range")
    
#26 CSV 파일 로드해서 데이터를 보내는 함수 - 절대적인 경로에서 데이터 파일을 가져오도록 구조 변경
@app.route('/deliveries')
def deliveries():
    selected_csv_file = select_csv_file()

    #41 Extract the filename from the selected file path
    # '/'로 자른 목록들 중 - 목록의 마지막 요소를 검색하는 목록 인덱싱
    csvFileName = selected_csv_file.split('/')[-1]
    return send_from_directory(BASE_PATH, csvFileName)

    # # CSV 파일 경로가 정확하고 서버 관점에서 액세스 가능한지 확인하도록!
    # return send_from_directory('C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static', 'deliveries.csv')

# //#44
@app.route('/drawroute')
def routes():
    selected_json_file = select_json_file()
    with open(selected_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return jsonify(data)


# # //#33 Driver 여러 명 페이지
# def load_deliveries():
#     return pd.read_csv("C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/deliveries.csv")

#41 현재시간 기준으로 배달 관련 csv 파일 가져오기
def load_deliveries():
    selected_csv_file = select_csv_file()
    return pd.read_csv(selected_csv_file)


#35 페이지 연결 
@app.route('/driver')
def client_home_page():
    return render_template('screen_Driver_home.html')

# //#33 Driver 여러 명 페이지
@app.route('/driver/<int:driver_id>')
def client_page(driver_id):    
    #33 URL에 포함된 driver_id가 csv파일(deliveries.csv)에 존재하는지 확인
    deliveries = load_deliveries()


    # #33 'render_template' 함수를 사용해 html 파일 렌더링 & 
    # # 'driver_id'를 HTML 파일로 전달하여, 페이지에서 사용할 수 있도록 함. 

    # delivery_info = deliveries[(deliveries['driver_id'] == driver_id) & (deliveries['delivery_type'] == '수거')]

    # # 'driver_id'를 바탕으로 deliveries.csv 파일에 있는 고객인지 확인 후 코드 실행

    if driver_id in deliveries['driver_id'].values:
        return render_template('screen_Driver.html', driverId = driver_id)
    else:
        abort(404)  # 존재하지 않는 client_id의 경우 404 에러 페이지를 반환


# @app.route('/')
# def index():
#     # return render_template('index.html')
#     return render_template('screen_Driver.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
    # app.run(debug=True)
