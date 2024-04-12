from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import pika

app = Flask(__name__)
socketio = SocketIO(app)

messages = []

def callback(ch, method, properties, body):
    message = body.decode()
    messages.append(message)
    socketio.emit('message', {'message': message}, namespace='/console')

@app.route('/')
def index():
    return render_template('index.html')

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
    socketio.run(app, debug=False)
