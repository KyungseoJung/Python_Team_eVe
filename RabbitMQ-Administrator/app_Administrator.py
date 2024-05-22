# // T-Map appkey = 5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu 

# //#26 관리자(Administrator) 웹화면 디자인 구성 
# //#27 관리자 웹페이지 디자인 
# //#28 수리모형 코드 통합 - [경로 계산하기] 버튼 누르면, 수리모형 코드 실행되도록 - csv 파일 지정한 파일 위치에 저장되도록

from flask import Flask, render_template
from flask import jsonify, request # //#28 수리모형 코드 통합을 위한 import
import pandas as pd
import eve_0522_test3 # //#28 수리모형 코드 통합 (Import 수리모형 함수를 포함한 Python file )

app = Flask(__name__)

@app.route('/calculate_path', methods=['POST'])
def calculate_path():
        # import eve_0521_test
    try:
        # //#28 내가 불러올 데이터
        datafile = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/orderData/S_02.txt"

        # //#28 내가 지정하는 경로 (파일 저장)
        pickle_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/all_k_shortest_paths.pickle_S_02"
        battery_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"
        truck_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"

        eve_0522_test3.solve(datafile, pickle_path, battery_csv_path, truck_csv_path)

        # Assuming there's a function in eve_0522_test.py to calculate the path and save a CSV
        # result = eve_0522_test.calculate_and_save()
        return jsonify({'success': True, 'message': '경로 계산 및 CSV 저장 완료!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

    
@app.route('/')
def administrator_page():
    return render_template('screen_Administrator.html')


if __name__ == '__main__':
    app.run(port=5004,debug=True)
