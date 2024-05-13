//#15 Flask를 거치지 않고, Node.js가 서버로 작동해 RabbitMQ로부터 데이터를 수신받는 코드를 구현(Node.js 설치 및 실행)
const express = require("express");
const http = require("http");
const socketIo = require("socket.io");
const amqp = require("amqplib/callback_api");

const app = express();
const server = http.createServer(app);
const io = socketIo(server);
const PORT = 3000;
const RABBITMQ_URL = "amqp://localhost";
const QUEUE = "realtime_location_queue1";

amqp.connect(RABBITMQ_URL, (error0, connection) => {
  if (error0) throw error0;
  connection.createChannel((error1, channel) => {
    if (error1) throw error1;

    channel.assertQueue(QUEUE, { durable: false });

    channel.consume(
      QUEUE,
      (msg) => {
        const message = msg.content.toString();
        console.log("Received: ", message);
        // Emitting message to all connected clients
        io.emit("new_message", { message });
      },
      { noAck: true }
    );
  });
});

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
