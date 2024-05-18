# //#12 (Driver 입장) 사용자의 실시간 위치를 html에서 얻어 Flask로 보내고, Flask에서는 실시간 위치 데이터를 RabbitMQ로 송신
# //#12 fix: (Driver 입장) 사용자의 실시간 위치를 html에서 얻어 (Flask를 거치지 않고) 바로 RabbitMQ로 송신 
# //#14 Flask를 거치지 않고, Node.js가 서버로 작동해 RabbitMQ로 데이터를 송신하는 코드를 구현(Node.js 설치 및 실행) 


from flask import Flask, render_template, request, redirect, url_for
import pika

app = Flask(__name__)

# 구조 설정
#12 fix: 주석처리 - (Driver 입장) 사용자의 실시간 위치를 html에서 얻어 (Flask를 거치지 않고) 바로 RabbitMQ로 송신
# exchange_name = ''     #7 데이터를 전송할 exchange의 이름을 설정
# queue_name = 'realtime_location_queue1'       #7 queue 이름 설정 - 수신할 때 필요***
# server_url = 'localhost'

#RabbitMQ 서버에 연결
#connection과 channel를 생성
#12 fix: 주석처리 - (Driver 입장) 사용자의 실시간 위치를 html에서 얻어 (Flask를 거치지 않고) 바로 RabbitMQ로 송신
# connection = pika.BlockingConnection(pika.ConnectionParameters(host=server_url))
# channel = connection.channel()

#queue_declare: channel를 통해 queue 선언(declare)
#12 fix: 주석처리 - (Driver 입장) 사용자의 실시간 위치를 html에서 얻어 (Flask를 거치지 않고) 바로 RabbitMQ로 송신
# channel.queue_declare(queue=queue_name)

# 여기부터가 핵심 - 반응형 RabbitMQ 송신
# 웹페이지에서 데이터 입력 -> 버튼 눌러 -> 여기서 데이터 받아서 -> RabbitMQ로 송신 
# 웹 페이지 라우트
@app.route('/')
def index():
    # return render_template('index_mq_ver1_sendCurrentPos.html') # //#12
    # return render_template('index_mq_ver2_sendCurrentPos.html') # //#12 fix
    return render_template('index_mq_ver3_sendCurrentPos.html') # //#15


# 데이터를 받아서 RabbitMQ로 송신
#12 fix: 주석처리 - (Driver 입장) 사용자의 실시간 위치를 html에서 얻어 (Flask를 거치지 않고) 바로 RabbitMQ로 송신
# @app.route('/send_location', methods=['POST'])
# def sendData():
#     latitude = request.form['latitude']
#     longitude = request.form['longitude']
#     message = f'Latitude: {latitude}, Longitude: {longitude}'
#     channel.basic_publish(exchange=exchange_name, routing_key=queue_name, body=message)
#     return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, port=5001)  # 다른 포트(예: 5001)를 지정하여 실행

    # 나중엔 debut=True 문 없애야 함
