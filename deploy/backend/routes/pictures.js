const {
  analyzeSkinCheckViaPython,
  analyzeLesionViaPython,
} = require("../services/pythonAIClient");

const { randomUUID } = require("crypto");
const express = require("express");
const multer = require("multer");
const path = require("path");
const { acronymMap } = require("../services/firebaseUpload");

const {
  uploadImageToFirebase,
  createEntry,
  getEntriesForUser,
  createEntryRef,
  mapAnalysisToAiResult,
} = require("../services/firebaseUpload");

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

function toBool(value) {
  return value === true || value === "true" || value === 1 || value === "1";
}

function isRejectedNotSkin(analysis) {
  return (
    analysis?.status === "success" &&
    analysis?.result?.label === "REJECTED_NOT_SKIN"
  );
}

function normalizeAiToLegacy(analysis) {
  if (!analysis || typeof analysis !== "object") return analysis;

  if (analysis.result && analysis.meta) return analysis;

  const latencyMs = Number(analysis.latency_ms ?? 0);

  if (analysis.status === "rejected") {
    const gate = analysis.gatekeeper || {};
    return {
      status: "success",
      result: {
        label: "REJECTED_NOT_SKIN",
        confidence: Number(gate.score ?? 0),
        attributes: gate.probs ?? {},
        predicted_index: Number(gate.predicted_index ?? -1),
        gatekeeper: gate,
        reason: analysis.reason ?? "No Skin Found in the Image. Try again with a clearer image",
      },
      meta: {
        model: "Gatekeeper",
        latency_ms: latencyMs,
        warnings: [],
      },
    };
  }

  if (analysis.status === "success") {
    const derm = analysis.dermassist || {};
    const gate = analysis.gatekeeper || {};

    const mappedAttributes = Object.entries(derm.attributes)
      .map(([label, confidence]) => ({
        label: acronymMap[label] ?? label,
        confidence: Number(confidence),
      }));

    return {
      status: "success",
      result: {
        label: acronymMap[derm.label] ?? "UNKNOWN",
        confidence: Number(derm.confidence ?? 0),
        attributes: mappedAttributes,
        predicted_index: Number(derm.predicted_index ?? -1),
        gatekeeper: gate,
      },
      meta: {
        model: "DermAssistAIModules",
        latency_ms: latencyMs,
        warnings: [],
      },
    };
  }

  if (analysis.status === "error") {
    return {
      status: "error",
      error: {
        code: "MODEL_INFERENCE_FAILED",
        message: analysis.message ?? "Inference failed",
        retriable: false,
      },
      meta: {
        model: "DermAssistAIModules",
        latency_ms: latencyMs,
        warnings: [],
      },
    };
  }

  return analysis;
}

function normalizeSkinCheckResult(analysis) {
  if (!analysis || typeof analysis !== "object") return analysis;

  if (analysis.result && analysis.meta) return analysis;

  const latencyMs = Number(analysis.latency_ms ?? 0);
  const gate = analysis.gatekeeper || {};

  if (analysis.status === "rejected") {
    return {
      status: "success",
      result: {
        label: "NON_SKIN",
        confidence: Number(gate.score ?? 0),
        predicted_index: Number(gate.predicted_index ?? -1),
        attributes: gate.probs ?? {},
      },
      meta: {
        model: "Gatekeeper",
        latency_ms: latencyMs,
        warnings: [],
      },
    };
  }

  if (analysis.status === "success") {
    return {
      status: "success",
      result: {
        label: "SKIN",
        confidence: Number(gate.score ?? 0),
        predicted_index: Number(gate.predicted_index ?? -1),
        attributes: gate.probs ?? {},
      },
      meta: {
        model: "Gatekeeper",
        latency_ms: latencyMs,
        warnings: [],
      },
    };
  }

  if (analysis.status === "error") {
    return {
      status: "error",
      error: {
        code: "MODEL_INFERENCE_FAILED",
        message: analysis.message ?? "Inference failed",
        retriable: false,
      },
      meta: {
        model: "Gatekeeper",
        latency_ms: latencyMs,
        warnings: [],
      },
    };
  }

  return analysis;
}

function getUploadFields(req) {
  const title = (req.body.title || "").toString();
  const description = (req.body.description || "").toString();
  const userId = (req.body.userId || "demo-user").toString();
  const bodyLocation = (req.body.bodyLocation || "").toString();

  const parsedTempDelta = Number(req.body.tempDelta);
  const tempDelta =
    req.body.tempDelta !== undefined &&
      req.body.tempDelta !== "" &&
      Number.isFinite(parsedTempDelta)
      ? parsedTempDelta
      : null;

  const bypassSkinCheck = toBool(req.body.bypassSkinCheck);

  return {
    title,
    description,
    userId,
    bodyLocation,
    tempDelta,
    bypassSkinCheck,
  };
}

async function runSkinCheck(file) {
  const requestId = randomUUID();
  const rawAnalysis = await analyzeSkinCheckViaPython({
    file,
    requestId,
  });

  return {
    requestId,
    analysis: normalizeSkinCheckResult(rawAnalysis),
  };
}

async function runLesionAnalysis(file, { bypassSkinCheck = false } = {}) {
  const firstRequestId = randomUUID();

  const firstRawAnalysis = await analyzeLesionViaPython({
    file,
    requestId: firstRequestId,
    bypassSkinCheck,
    forceLesionOnly: bypassSkinCheck,
  });

  let analysis = normalizeAiToLegacy(firstRawAnalysis);

  // Edge case coverage:
  // if bypass was requested, but backend still somehow returns rejected,
  // retry straight to lesion-only again with hard force.
  if (!(bypassSkinCheck && isRejectedNotSkin(analysis))) {
    return {
      requestId: firstRequestId,
      analysis,
      retried: false,
    };
  }

  const retryRequestId = randomUUID();

  const retryRawAnalysis = await analyzeLesionViaPython({
    file,
    requestId: retryRequestId,
    bypassSkinCheck: true,
    forceLesionOnly: true,
  });

  analysis = normalizeAiToLegacy(retryRawAnalysis);

  return {
    requestId: retryRequestId,
    analysis,
    retried: true,
  };
}

router.post("/skin-check", upload.single("image"), async (req, res) => {
  console.log("Incoming request: POST /uploads/skin-check");

  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  try {
    const { analysis, requestId } = await runSkinCheck(req.file);

    return res.status(200).json({
      message: "Skin check successful",
      requestId,
      analysis,
    });
  } catch (err) {
    console.error("Skin check failed:", err.message);
    return res.status(err.httpStatus || 502).json({
      error: "Skin check failed",
      details: err.message,
      requestId: err.requestId || null,
      body: err.body || null,
    });
  }
});

router.post("/image", upload.single("image"), async (req, res) => {
  console.log("Incoming request: POST /uploads/image");

  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const {
    title,
    description,
    userId,
    bodyLocation,
    tempDelta,
    bypassSkinCheck,
  } = getUploadFields(req);

  try {
    const ext = path.extname(req.file.originalname) || ".jpg";

    const entryRef = createEntryRef(userId);
    const entryId = entryRef.id;

    const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}${ext}`;
    const storagePath = `uploads/${userId}/entries/${entryId}/${filename}`;

    console.log("Uploading image to Firebase...");
    const firebaseResult = await uploadImageToFirebase(req.file, {
      storagePath,
      metadata: { userId, entryId },
    });

    let result = null;
    let analysis = null;
    let analysisError = null;
    let aiResult = null;

    try {
      result = await runLesionAnalysis(req.file, { bypassSkinCheck });
      analysis = result.analysis;
      aiResult = mapAnalysisToAiResult(analysis);

      console.log(
        "Python lesion analysis:",
        analysis?.status,
        "label=",
        acronymMap[analysis?.result?.label] ?? analysis?.result?.label,
        "confidence=",
        analysis?.result?.confidence,
        "bypassSkinCheck=",
        bypassSkinCheck,
        "retried=",
        result.retried,
      );
    } catch (err) {
      console.error("Python lesion analysis failed:", err.message);
      analysisError = {
        message: err.message,
        httpStatus: err.httpStatus || 502,
        requestId: err.requestId || null,
        body: err.body || null,
      };
    }

    const entry = await createEntry({
      userId,
      entryId,
      title,
      description,
      storagePath: firebaseResult.storagePath,
      bodyLocation,
      tempDelta,
      aiResult,
    });

    return res.status(200).json({
      message: "Upload successful",
      entry,
      firebase: firebaseResult,
      analysis,
      analysisError,
      bypassSkinCheck,
      retriedLesionAnalysis: result?.retried ?? false,
    });
  } catch (err) {
    console.error("Upload failed:", err);
    return res.status(500).json({ error: "Upload failed" });
  }
});

router.get("/images", async (req, res) => {
  const userId = (req.query.userId || "demo-user").toString();
  const limit = Math.min(parseInt(req.query.limit || "50", 10), 200);

  console.log("Incoming request: GET /uploads/images");

  try {
    const entries = await getEntriesForUser({ userId, limit });
    console.log("Completed request: GET /uploads/images -> 200");
    return res.status(200).json({ entries });
  } catch (err) {
    console.error("Failed to fetch entries:", err);
    return res.status(500).json({ error: "Failed to fetch entries" });
  }
});

module.exports = router;
