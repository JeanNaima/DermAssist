const axios = require("axios");
const { randomUUID } = require("crypto");

const PYTHON_API_BASE = process.env.PYTHON_API_BASE || "http://127.0.0.1:8000";


async function analyzeImageViaPython({ userId, imageUrl, requestId }) {
  const rid = requestId || randomUUID();

  const payload = {
    request_id: rid,
    user_id: userId,
    image: {
      type: "url",
      url: imageUrl,
    },
  };

  try {
    const resp = await axios.post(`${PYTHON_API_BASE}/analyzeimage`, payload, {
      headers: {
        "Content-Type": "application/json",
        "x-request-id": rid,
      },
      timeout: 60_000, 
      validateStatus: () => true, 
    });

    if (resp.status >= 200 && resp.status < 300) {
      return resp.data;
    }

    const message =
      resp.data?.error?.message ||
      resp.data?.message ||
      `Python service returned HTTP ${resp.status}`;

    const err = new Error(message);
    err.httpStatus = resp.status;
    err.body = resp.data;
    err.requestId = rid;
    throw err;
  } catch (e) {
    const err = new Error(e.message || "Failed calling Python service");
    err.requestId = requestId;
    err.cause = e;

    if (e.response) {
      err.httpStatus = e.response.status;
      err.body = e.response.data;
    }
    throw err;
  }
}

module.exports = { analyzeImageViaPython };
