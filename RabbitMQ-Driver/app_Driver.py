from flask import Flask, render_template, send_from_directory
import pandas as pd

app = Flask(__name__)

#26 CSV 파일 로드해서 데이터를 보내는 함수 - 절대적인 경로에서 데이터 파일을 가져오도록 구조 변경
@app.route('/deliveries')
def deliveries():
    # CSV 파일 경로가 정확하고 서버 관점에서 액세스 가능한지 확인하도록!
    return send_from_directory('C:/GitStudy/Python_Team_eVe/RabbitMQ-Driver/static', 'deliveries.csv')

@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('screen_Driver.html')

if __name__ == '__main__':
    app.run(debug=True)
