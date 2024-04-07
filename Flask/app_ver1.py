#2 웹페이지에서 메시지 수신하기

from flask import Flask, render_template
import pika
import threading    # //#3 무한루프 오류 방지

app = Flask(__name__)

messages = []           # //#3 무한루프 오류 방지 
lock = threading.Lock() # //#3 무한루프 오류 방지

def callback(ch, method, properties, body): # //#3
    with lock:
        messages.append(body.decode())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/console')
def console():
    # messages = []

    # def callback(ch, method, properties, body):
    #     messages.append(body.decode())

    # connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    # channel = connection.channel()

    # channel.queue_declare(queue='hello')

    # channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    # channel.start_consuming()
    # connection.close()

    return render_template('console.html', messages=messages)


def consume_messages():  # //#3 무한루프 오류 방지 - 메시지 수신하기 위한 코드
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()
    connection.close()

if __name__ == '__main__':
    threading.Thread(target = consume_messages, daemon=True).start()    # //#3

    # app.run(debug=True)
    app.run(debug=False)
    
