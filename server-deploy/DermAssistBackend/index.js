const express = require("express");
const cors = require("cors");

const picturesRouter = require("./routes/pictures");
const authRoutes = require("./routes/authRoutes");

const app = express();

app.set("trust proxy", 1);

app.use(cors());
app.use(express.json());

app.use("/uploads", picturesRouter);
app.use("/", authRoutes);

//Dummy endpoint
app.get("/dummy", (req, res) => {
  res.json({ message: "Good Job" });
});

app.get("/healthz", (req, res) => res.json({ status: "ok" }));

app.use((err, req, res, next) => {
  console.error(err);

  if (res.headersSent) {
    return next(err);
  }

  const status = err.statusCode || err.status || 500;
  const message = err.publicMessage || err.message || "Something went wrong!";
  return res.status(status).json({ error: message });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Backend running on port ${PORT}`);
});