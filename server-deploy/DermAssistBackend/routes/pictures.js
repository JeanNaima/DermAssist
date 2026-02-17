const { analyzeImageViaPython } = require("../services/pythonAIClient");
const { randomUUID } = require("crypto");
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || "http://127.0.0.1:8000";
const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const {
  uploadImageToFirebase,
  createEntry,
  getEntriesForUser,
} = require("../services/firebaseUpload");

const router = express.Router();

const uploadDir = path.join(__dirname, "..", "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB image size limit for now
});

router.post("/image", upload.single("image"), async (req, res) => {
  console.log("Incoming request: POST /uploads/image");

  if (!req.file) {
    console.log("Error: No file received");
    return res.status(400).json({ error: "No file uploaded" });
  }
  const title = (req.body.title || "").toString();
  const description = (req.body.description || "").toString();
  const userId = (req.body.userId || "demo-user").toString();
  const bodyLocation = (req.body.bodyLocation || "").toString();

  try {
    console.log("File received successfully");
    console.log("Original filename:", req.file.originalname);
    console.log("Mimetype:", req.file.mimetype);
    console.log("File size (bytes):", req.file.size);
    console.log("Title:", title);
    console.log("Description length:", description.length);

    const ext = path.extname(req.file.originalname) || ".jpg";
    const filename = `${Date.now()}-${Math.round(Math.random() * 1e9)}${ext}`;
    const localPath = path.join(uploadDir, filename);

    //Uploading to Firebase----------------------------
    console.log("Uploading image to Firebase...");
    const storagePath = `uploads/${userId}/${filename}`;
    const firebaseResult = await uploadImageToFirebase(req.file, {
      storagePath,
    });
    const publicImageUrl = firebaseResult.url;
    console.log("Uploaded to Firebase at:", firebaseResult.storagePath);

    //Create Firestore entry
    const entry = await createEntry({
      userId,
      title,
      description,
      storagePath: firebaseResult.storagePath,
      bodyLocation: bodyLocation,
    });

    let analysis = null;
    let analysisError = null;

    if (publicImageUrl && publicImageUrl.startsWith("http")) {
      const requestId = randomUUID(); 
      try {
        analysis = await analyzeImageViaPython({
          userId,
          imageUrl: publicImageUrl,
          requestId,
        });
        console.log("Python analysis success:", analysis?.result?.label);
      } catch (err) {
        console.error("Python analysis failed:", err.message);
        analysisError = {
          message: err.message,
          httpStatus: err.httpStatus || 502,
          requestId: err.requestId || requestId,
          body: err.body || null,
        };
      }
    }
    return res.json({
      message: "Upload successful",
      entry,
      firebase: firebaseResult,
      analysis,
      analysisError,
      local: {
        filename,
        path: localPath,
        size: req.file.size,
      },
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
