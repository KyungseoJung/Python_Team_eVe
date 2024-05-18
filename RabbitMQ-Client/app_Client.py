'''
# 20 screen_Client.html 파일 실행하기 위함
#22 고객의 수만큼 다른 웹페이지 실행되도록 & client_id에 따라 다른 RabbitMQ 주소를 부여받고, 그에 대한 데이터를 수신받도록
# 고민: 다른 URL로 연결만 하면 되나 or 다른 port 번호를 가져야 하나 -> 그냥 다른 url로 연결하면 될 듯
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

    #22 'render_template' 함수를 사용해 html 파일 렌더링 & 
    # 'client_id'를 HTML 파일로 전달하여, 페이지에서 사용할 수 있도록 함. 
    if client_id in deliveries['client_id'].values:
        return render_template('screen_Client.html', clientId=client_id)
    # client_id 전달: - 공부 필요
    else:
        abort(404)  # 존재하지 않는 client_id의 경우 404 에러 페이지를 반환

if __name__ == "__main__":
    app.run(port=5002,debug=True)

    # 나중엔 debut=True 문 없애야 함

