const axios = require("axios");
const FormData = require("form-data");

const AI_API_BASE_URL = process.env.AI_API_BASE_URL || "http://ai-api:8000";

function boolField(value) {
  return value ? "true" : "false";
}

function buildImageForm(file, extraFields = {}) {
  if (!file || !file.buffer) {
    const err = new Error("Missing file.buffer for Python analysis");
    err.httpStatus = 400;
    throw err;
  }

  const form = new FormData();

  form.append("image", file.buffer, {
    filename: file.originalname || "image.jpg",
    contentType: file.mimetype || "image/jpeg",
    knownLength: file.size,
  });

  for (const [key, value] of Object.entries(extraFields)) {
    form.append(
      key,
      typeof value === "boolean" ? boolField(value) : String(value),
    );
  }

  return form;
}

async function postImageToPython(routePath, { file, requestId, fields = {} }) {
  const form = buildImageForm(file, fields);

  try {
    const resp = await axios.post(`${AI_API_BASE_URL}${routePath}`, form, {
      headers: {
        ...form.getHeaders(),
        ...(requestId ? { "x-request-id": String(requestId) } : {}),
      },
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 120_000,
      validateStatus: () => true,
    });

    if (resp.status >= 200 && resp.status < 300) {
      return resp.data;
    }

    const message =
      resp.data?.message ||
      resp.data?.error?.message ||
      `Python service returned HTTP ${resp.status}`;

    const err = new Error(message);
    err.httpStatus = resp.status;
    err.body = resp.data;
    err.requestId = requestId || null;
    throw err;
  } catch (e) {
    const err = new Error(e.message || "Failed calling Python service");
    err.requestId = requestId || null;
    err.cause = e;

    if (e.response) {
      err.httpStatus = e.response.status;
      err.body = e.response.data;
    } else if (e.httpStatus) {
      err.httpStatus = e.httpStatus;
      err.body = e.body;
    } else {
      err.httpStatus = 502;
    }

    throw err;
  }
}

async function analyzeSkinCheckViaPython({ file, requestId }) {
  return postImageToPython("/skin-check", {
    file,
    requestId,
  });
}

async function analyzeLesionViaPython({
  file,
  requestId,
  bypassSkinCheck = false,
  forceLesionOnly = false,
}) {
  return postImageToPython("/lesion-detection", {
    file,
    requestId,
    fields: {
      bypassSkinCheck,
      forceLesionOnly,
    },
  });
}

module.exports = {
  analyzeSkinCheckViaPython,
  analyzeLesionViaPython,
};
