'''

#35 고객 화면과 Driver 화면에서 코드 추가 - 모바일 웹 화면으로 확인할 때, 페이지 연결 및 구조가 적절히 보이도록

'''
from flask import Flask, render_template, send_from_directory, abort
import pandas as pd

app = Flask(__name__)

#26 CSV 파일 로드해서 데이터를 보내는 함수 - 절대적인 경로에서 데이터 파일을 가져오도록 구조 변경
@app.route('/deliveries')
def deliveries():
    # CSV 파일 경로가 정확하고 서버 관점에서 액세스 가능한지 확인하도록!
    return send_from_directory('C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static', 'deliveries.csv')

# //#33 Driver 여러 명 페이지
def load_deliveries():
    return pd.read_csv("C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/deliveries.csv")

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
    app.run(debug=True)
