# from flask import Flask, render_template
# import pika

# app = Flask(__name__)

# # RabbitMQ 서버 설정
# server_url = 'localhost'
# queue_name = 'hello'

# # 웹 페이지에 메시지를 표시하는 함수
# @app.route('/')
# def index():
#     return render_template('index.html', message="No message received yet")

# # RabbitMQ에서 메시지를 수신하여 웹 페이지에 표시하는 함수
# def consume_from_rabbitmq():
#     # RabbitMQ 서버에 연결
#     connection = pika.BlockingConnection(pika.ConnectionParameters(host=server_url))
#     channel = connection.channel()

#     # Queue 선언
#     channel.queue_declare(queue=queue_name)

#     # 콜백 함수: 메시지를 수신하면 웹 페이지에 표시
#     def callback(ch, method, properties, body):
#         message = body.decode('utf-8')
#         print(f"Received message from RabbitMQ: {message}")
#         # 메시지를 템플릿에 전달하여 웹 페이지에 표시
#         app.config['MESSAGE'] = message

#     # RabbitMQ 큐에서 메시지 소비
#     channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

#     # 메시지 대기
#     print(' [*] Waiting for messages from RabbitMQ. To exit press CTRL+C')
#     channel.start_consuming()

# # 웹 서버 실행
# if __name__ == '__main__':
#     # RabbitMQ에서 메시지를 수신하여 웹 페이지에 표시하는 함수를 실행 (스레드로 실행)
#     consume_thread = Thread(target=consume_from_rabbitmq)
#     consume_thread.start()
    
#     # Flask 웹 서버 실행
#     app.run(debug=True)
