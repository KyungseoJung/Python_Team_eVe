# //#4 Producer_string.ipynb 파일에서 RabbitMQ로 송신하는 메시지를 실시간으로 수신해서 웹페이지에 보이도록 하기(단, 새로고침 없이!!)

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pika

app = Flask(__name__)       # Flask 애플리케이션 생성
socketio = SocketIO(app)    # Socket.IO 객체 생성

messages = []               # 받은 메시지를 저장할 리스트

# RabbitMQ로부터 메시지를 수신하는 함수
def callback(ch, method, properties, body):
    messages.append(body.decode())  # 받은 메세지를 messages 리스트에 추가
    # 새로운 메시지를 클라이언트에게 전송
    socketio.emit('new_message', {'message': body.decode()}, namespace='/')  # 수정된 부분: 네임스페이스 추가

# 루트 URL에 대한 라우팅 함수: index.html을 렌더링하고, 메시지들을 함께 전달
@app.route('/')
def index():
    return render_template('index.html', messages=messages)  # 수정된 부분: index.html로 메시지들을 함께 전달

# /console URL에 대한 라우팅 함수: console.html을 렌더링하고, 메시지들을 함께 전달
@app.route('/console')
def console():
    return render_template('console.html', messages=messages)

# RabbitMQ에서 메시지를 소비하고, 수신한 메시지를 처리하는 함수
def consume_messages():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
    connection.close()

# 메인 모듈일 때만 서버 실행
if __name__ == '__main__':
    socketio.start_background_task(consume_messages)    # 백그라운드에서 RabbitMQ에서 메시지 수신하는 함수 실행
    socketio.run(app)   # Flask 애플리케이션 실행
