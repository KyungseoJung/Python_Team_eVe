//#14 Flask를 거치지 않고, Node.js가 서버로 작동해 RabbitMQ로 데이터를 송신하는 코드를 구현(Node.js 설치 및 실행)
/*
#14 server.js 파일은 Node.js 환경에서 RabbitMQ와 통신하는 웹 서버를 구축하는 스크립트
이 스크립트는 위치 데이터를 실시간으로 수신하고 RabbitMQ로 전송하는 역할.
전체적인 시스템에서 중앙 서버 역할을 수행. 
데이터가 RabbitMQ로 정확히 전송되어 처리될 수 있도록 보장하는 중요한 기능을 담당
*/

// #14 1) 모듈 임포트:
/*
express: Node.js의 웹 서버 기능을 구축하기 위한 프레임워크입니다.
cors: Cross-Origin Resource Sharing을 가능하게 하는 미들웨어로, 서버가 다른 도메인/포트에서 동작하는 클라이언트의 요청을 수락하도록 설정합니다.
bodyParser: 요청에서 JSON 데이터를 파싱하기 위한 미들웨어입니다.
amqp: RabbitMQ 서버와의 연결 및 통신을 위한 라이브러리입니다.
*/
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const amqp = require("amqplib/callback_api");

// #14 2) 서버 및 상수 설정:
/*
app: Express 애플리케이션의 인스턴스를 생성합니다.
PORT: 서버가 수신할 포트 번호입니다.
RABBITMQ_URL: RabbitMQ 서버의 URL입니다.
QUEUE: 메시지를 보낼 RabbitMQ의 큐 이름입니다. 
*/
const app = express();
const PORT = 3000;
const RABBITMQ_URL = "amqp://localhost"; // Modify as needed
const QUEUE = "realtime_location_queue1";

// #14 3) 미들웨어 설정:
// 모든 라우트에서 CORS를 활성화하고, 들어오는 요청의 본문에서 JSON 데이터를 파싱
app.use(cors()); // Enable CORS for all routes
app.use(bodyParser.json());

// #14 4) RabbitMQ 연결 및 채널 설정:
/*
RabbitMQ 서버에 연결을 시도하고, 성공적으로 연결되면 채널을 생성.
채널은 메시지를 송수신하는 데 사용.
*/
amqp.connect(RABBITMQ_URL, (error0, connection) => {
  if (error0) {
    throw error0;
  }
  connection.createChannel((error1, channel) => {
    if (error1) {
      throw error1;
    }

    //#14 5) 위치 데이터 수신 및 RabbitMQ로 송신:
    /*
    - 수신: /send_location 엔드포인트를 통해 클라이언트로부터 위도와 경도 데이터를 받습니다.
    - 송신: 받은 데이터를 메시지로 구성하고, RabbitMQ의 지정된 큐에 메시지를 보냅니다.
    메시지를 보내고, 클라이언트에게 성공적으로 전송되었다는 응답을 보냅니다.
    */
    app.post("/send_location", (req, res) => {
      const { latitude, longitude } = req.body;
      const message = `Latitude: ${latitude}, Longitude: ${longitude}`;
      channel.assertQueue(QUEUE, { durable: false });
      channel.sendToQueue(QUEUE, Buffer.from(message));
      console.log("Sent: ", message);
      res.send("Location sent to RabbitMQ");
    });

    //#14 6) 서버 실행:
    // 정의된 포트에서 서버를 실행하고, 서버가 정상적으로 실행되고 있음을 로그로 출력.
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  });
});
