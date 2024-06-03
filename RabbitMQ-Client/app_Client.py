'''
# 20 screen_Client.html 파일 실행하기 위함
#22 고객의 수만큼 다른 웹페이지 실행되도록 & client_id에 따라 다른 RabbitMQ 주소를 부여받고, 그에 대한 데이터를 수신받도록
# 고민: 다른 URL로 연결만 하면 되나 or 다른 port 번호를 가져야 하나 -> 그냥 다른 url로 연결하면 될 듯
#25-2 고객 화면 - "서비스 시작 시간"에 실시간 RabbitMQ로 "배달완료"메시지 왔을 때 시간 넣기,
  "예상 서비스 완료 시간"에 deliveries.csv에 있는 time 값 넣기(단, 해당 client_id에 해당하면서 delivery_type이 "배달"일 때 행에 해당하는 time값)
#34 고객이 보는 지도 - 고객 주문 위치 받아서 마커 표시 - 고객 주문 위치 중심으로 지도 고정

#35 고객 화면과 Driver 화면에서 코드 추가 - 모바일 웹 화면으로 확인할 때, 페이지 연결 및 구조가 적절히 보이도록
#36 고객 화면 - 예약 화면 추가
#37 고객 화면 - 주문 정보 데이터 RabbitMQ로 송신하기 && 다음 페이지(screen_Client.html)로 보내기

#41 현재시간 기준으로 배달 관련 csv 파일 3가지 중 하나를 가져오기
'''

'''
#60
cmd 창에 ipconfig를 통해 IPv4 주소를 찾아

1) 
모바일로 실행 시,
screen_Client_home.html의 url 주소는 IPv4 주소를 작성하도록
-> 매일 수정되니까 실행할 때마다 확인 - 수정 - 실행
-> app_Client.py 파일의 port 번호와 screen_Client_home.html 의 port번호를 동일하게 맞추기

2) RabbitMQ 주소 맞추기
var ws = new WebSocket("ws://127.0.0.1:15674/ws");  -> 로컬로 연결되어 있는 것
IPv4 주소를 찾아서 아래와 같이 수정해주기
예를 들어, 192.168.50.178이라면, var ws = new WebSocket("ws://192.168.50.178:15674/ws");
'''

from flask import Flask, render_template, abort, request, redirect, url_for
import pandas as pd
from datetime import datetime   #41


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

#22 CSV 파일 로드해서 데이터를 읽어오는 함수
#41 현재시간 기준으로 배달 관련 csv 파일 가져오기
def load_deliveries():
    selected_csv_file = select_csv_file()
    return pd.read_csv(selected_csv_file)


#35 페이지 연결 
@app.route('/client')
def client_home_page():
    return render_template('screen_Client_home.html')

#36 고객 화면 - 예약 화면 추가
@app.route('/client/<clientId>/order')
def client_order(clientId):
    return render_template('screen_Client_order.html', clientId=clientId)


@app.route('/client/<int:client_id>')
def client_page(client_id):    
    #22 URL에 포함된 client_id가 csv파일(Truck_routes_E_0n.csv)에 존재하는지 확인
    deliveries = load_deliveries()


    # #22 'render_template' 함수를 사용해 html 파일 렌더링 & 
    # # 'client_id'를 HTML 파일로 전달하여, 페이지에서 사용할 수 있도록 함. 
    # # 'client_id'를 바탕으로 Truck_routes_E_0n.csv 파일에 있는 고객인지 확인 후 코드 실행
    # if client_id in deliveries['client_id'].values:
    #     return render_template('screen_Client.html', clientId=client_id)
    # else:
    #     abort(404)  # 존재하지 않는 client_id의 경우 404 에러 페이지를 반환

    #25-2 예정 서비스 완료 시간 (=수거 완료 시간) 표시하기 - 기존 #22 코드를 여기에 병합
    # 특정 client_id에 해당하고, delivery_type이 "수거"인 행에 해당하는 time 가져오기
    serviceComplete_info = deliveries[(deliveries['client_id'] == client_id) & (deliveries['delivery_type'] == '수거')]

    #23 fix: 각 고객은 자신에게 할당된 Driver위치가 보이도록
    serviceStart_info = deliveries[(deliveries['client_id'] == client_id) & (deliveries['delivery_type'] == '배달')]    

    # 37 app_Client.py 파일에서 screen_Client_order.html에서 데이터를 받고, screen_Client.html로 전달하는 기능을 추가
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    charge_amount = request.args.get('charge_amount')
    reservation_time = request.args.get('reservation_time')
    car_number = request.args.get('car_number')

    # 위에서 찾은 serviceComplete_info에 해당하는 데이터가 존재한다면, 아래 코드 실행
    if not serviceComplete_info.empty:
        serviceComplete_time = serviceComplete_info.iloc[0]['time']  # 처음으로 일치하는 값이 우리가 필요한 데이터일 테니까. (물론, 중복된 데이터가 존재해서도 안 됨.)
        
        #23 fix: 각 고객은 자신에게 할당된 Driver위치가 보이도록
        if not serviceStart_info.empty:
            serviceStart_driver = serviceStart_info.iloc[0]['driver_id']

            #34 고객이 보는 지도 - 고객 주문 위치 받아서 마커 표시 - 고객 주문 위치 중심으로 지도 고정
            servicePos_Lat = serviceStart_info.iloc[0]['latitude']
            servicePos_Lng = serviceStart_info.iloc[0]['longitude']

            # 'render_template'함수를 이용해 'client_id'와 'serviceComplete_time' 데이터를 HTML 파일로 전달하여, 페이지에서 사용할 수 있도록 함.
            #23 fix: 각 고객은 자신에게 할당된 Driver위치가 보이도록 // #34
            # 37 app_Client.py 파일에서 screen_Client_order.html에서 데이터를 받고, screen_Client.html로 전달하는 기능을 추가
            return render_template('screen_Client.html', clientId=client_id, serviceCompleteTime=serviceComplete_time, serviceStartDriver = serviceStart_driver, servicePosLat=servicePos_Lat, servicePosLng = servicePos_Lng,
                                   latitude=latitude, longitude = longitude, chargeAmount = charge_amount, reservationTime = reservation_time, carNumber = car_number)     
    else:
        abort(404)  # 특정 "client_id"에서 "delivery_type"의 값이 '배달'이 없는 경우 에러 화면 표시


if __name__ == "__main__":
    # app.run(port=5002,debug=True)
    app.run(host="0.0.0.0", port=5002,debug=True)

    # 나중엔 debut=True 문 없애야 함

