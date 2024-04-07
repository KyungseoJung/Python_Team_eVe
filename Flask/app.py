# //#4 Producer_string.ipynb 파일에서 RabbitMQ로 송신하는 메시지를 실시간으로 수신해서 웹페이지에 보이도록 하기(단, 새로고침 없이!!)

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pika

app = Flask(__name__)
socketio = SocketIO(app)

messages = []

def callback(ch, method, properties, body):
    messages.append(body.decode())
    socketio.emit('new_message', {'message': body.decode()}, namespace='/')  # 수정된 부분: 네임스페이스 추가

@app.route('/')
def index():
    return render_template('index.html', messages=messages)  # 수정된 부분: index.html로 메시지들을 함께 전달

@app.route('/console')
def console():
    return render_template('console.html', messages=messages)

def consume_messages():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
    connection.close()

if __name__ == '__main__':
    socketio.start_background_task(consume_messages)
    socketio.run(app)
