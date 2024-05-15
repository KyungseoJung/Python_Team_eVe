# //#15 Flask를 거치지 않고, Node.js가 서버로 작동해 RabbitMQ로부터 데이터를 수신받는 코드를 구현(Node.js 설치 및 실행)

from flask import Flask, render_template

app = Flask(__name__)   # Flask 애플리케이션 생성

# 루트 URL에 대한 라우팅 함수: index_mq_ver2_receiveCurrentPosMap.html을 렌더링

@app.route('/')
def index():
    return render_template('index_mq_ver5_receiveCurrentPosMap.html')  
# 'index.html'은 templates 폴더 내부에 위치해야 함.

if __name__ == '__main__':
    app.run(port=5002)  # Flask 애플리케이션을 5002번 포트에서 실행
