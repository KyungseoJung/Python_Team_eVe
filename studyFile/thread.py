import threading
import subprocess

if __name__ == "__main__":
    # Producer 코드 실행 명령어
    producer_command = ["python", "RabbitMQ-Driver/Producer.py"]

    # app_mq_ver1_receiveCurrentPos 코드 실행 명령어
    receiver_command = ["python", "RabbitMQ-Client/Consumer_mq_receiveCurrentPos.py"]

    # Producer 코드를 실행하는 프로세스 생성
    producer_process = subprocess.Popen(producer_command)

    # app_mq_ver1_receiveCurrentPos 코드를 실행하는 프로세스 생성
    receiver_process = subprocess.Popen(receiver_command)

    # 두 프로세스가 종료될 때까지 대기
    producer_process.wait()
    receiver_process.wait()
