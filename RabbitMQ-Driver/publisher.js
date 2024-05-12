// publisher.js
const http = require("http");
const url = require("url");
const pika = require("amqplib/callback_api");

const server_url = "localhost"; // RabbitMQ 서버 URL
const exchange_name = ""; // 교환기 이름 설정, 필요한 경우 수정
const queue_name = "realtime_location_queue1"; // 큐 이름

// RabbitMQ 서버에 연결
pika.connect(`amqp://${server_url}`, function (error0, connection) {
  if (error0) {
    throw error0;
  }
  connection.createChannel(function (error1, channel) {
    if (error1) {
      throw error1;
    }

    channel.assertQueue(queue_name, {
      durable: false,
    });

    // HTTP 서버 생성
    http
      .createServer((req, res) => {
        const queryObject = url.parse(req.url, true).query;
        const latitude = queryObject.latitude;
        const longitude = queryObject.longitude;

        if (latitude && longitude) {
          const message = `Latitude: ${latitude}, Longitude: ${longitude}`;
          channel.sendToQueue(queue_name, Buffer.from(message));
          console.log("Sent:", message);
        }

        res.writeHead(200, {
          "Content-Type": "text/plain",
          "Access-Control-Allow-Origin": "*", // CORS 정책 허용
        });
        res.end("Location sent to RabbitMQ\n");
      })
      .listen(3000, () => {
        console.log("Server listening on port 3000");
      });
  });
});
