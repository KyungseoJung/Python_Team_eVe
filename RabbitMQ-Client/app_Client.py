# 20 screen_Client.html 파일 실행하기 위함
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('screen2_Client.html')

if __name__ == '__main__':
    app.run(port=5002, debug=True)
    
    # 나중엔 debut=True 문 없애야 함
