'''
# 20 screen_Client.html 파일 실행하기 위함
#22 고객의 수만큼 다른 웹페이지 실행되도록 & client_id에 따라 다른 RabbitMQ 주소를 부여받고, 그에 대한 데이터를 수신받도록
# 고민: 다른 URL로 연결만 하면 되나 or 다른 port 번호를 가져야 하나 -> 그냥 다른 url로 연결하면 될 듯
#25-2 고객 화면 - "서비스 시작 시간"에 실시간 RabbitMQ로 "배달완료"메시지 왔을 때 시간 넣기,
  "예상 서비스 완료 시간"에 deliveries.csv에 있는 time 값 넣기(단, 해당 client_id에 해당하면서 delivery_type이 "배달"일 때 행에 해당하는 time값)

'''
from flask import Flask, render_template, abort
import pandas as pd

app = Flask(__name__)

#22 CSV 파일 로드해서 데이터를 읽어오는 함수
def load_deliveries():
    return pd.read_csv("C:/GitStudy/Python_Team_eVe/RabbitMQ-Driver/static/deliveries.csv")

@app.route('/client/<int:client_id>')
def client_page(client_id):    
    #22 URL에 포함된 client_id가 csv파일(deliveries.csv)에 존재하는지 확인
    deliveries = load_deliveries()


    # #22 'render_template' 함수를 사용해 html 파일 렌더링 & 
    # # 'client_id'를 HTML 파일로 전달하여, 페이지에서 사용할 수 있도록 함. 
    # # 'client_id'를 바탕으로 deliveries.csv 파일에 있는 고객인지 확인 후 코드 실행
    # if client_id in deliveries['client_id'].values:
    #     return render_template('screen_Client.html', clientId=client_id)
    # else:
    #     abort(404)  # 존재하지 않는 client_id의 경우 404 에러 페이지를 반환

    #25-2 예정 서비스 완료 시간 (=수거 완료 시간) 표시하기 - 기존 #22 코드를 여기에 병합
    # 특정 client_id에 해당하고, delivery_type이 "수거"인 행에 해당하는 time 가져오기
    delivery_info = deliveries[(deliveries['client_id'] == client_id) & (deliveries['delivery_type'] == '수거')]

    # 위에서 찾은 delivery_info에 해당하는 데이터가 존재한다면, 아래 코드 실행
    if not delivery_info.empty:
        delivery_time = delivery_info.iloc[0]['time']  # 처음으로 일치하는 값이 우리가 필요한 데이터일 테니까. (물론, 중복된 데이터가 존재해서도 안 됨.)
        # 'render_template'함수를 이용해 'client_id'와 'delivery_time' 데이터를 HTML 파일로 전달하여, 페이지에서 사용할 수 있도록 함.
        return render_template('screen_Client.html', clientId=client_id, deliveryTime=delivery_time)     
    else:
        abort(404)  # 특정 "client_id"에서 "delivery_type"의 값이 '배달'이 없는 경우 에러 화면 표시

if __name__ == "__main__":
    app.run(port=5002,debug=True)

    # 나중엔 debut=True 문 없애야 함

