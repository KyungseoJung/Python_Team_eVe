const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const amqp = require("amqplib/callback_api");

const app = express();
const PORT = 3000;
const RABBITMQ_URL = "amqp://localhost"; // Modify as needed
const QUEUE = "realtime_location_queue1";

app.use(cors()); // Enable CORS for all routes
app.use(bodyParser.json());

amqp.connect(RABBITMQ_URL, (error0, connection) => {
  if (error0) {
    throw error0;
  }
  connection.createChannel((error1, channel) => {
    if (error1) {
      throw error1;
    }

    app.post("/send_location", (req, res) => {
      const { latitude, longitude } = req.body;
      const message = `Latitude: ${latitude}, Longitude: ${longitude}`;
      channel.assertQueue(QUEUE, { durable: false });
      channel.sendToQueue(QUEUE, Buffer.from(message));
      console.log("Sent: ", message);
      res.send("Location sent to RabbitMQ");
    });

    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  });
});
