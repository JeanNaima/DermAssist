const path = require("path");
const { bucket, db } = require("../config/firebase");

async function createEntry({ userId, title, description, storagePath }) {
  const createdAt = Date.now();

  const docRef = await db.collection("entries").add({
    userId,
    title,
    description,
    storagePath,
    createdAt,
    shared: false,
  });

  return {
    id: docRef.id,
    userId,
    title,
    description,
    storagePath,
    createdAt,
    shared: false,
  };
}

async function uploadImageToFirebase(file, options = {}) {
  if (!file || !file.buffer) {
    throw new Error("Invalid file during upload");
  }

  const ext = path.extname(file.originalname) || ".jpg";
  const filename =
    options.filename ||
    `${Date.now()}-${Math.round(Math.random() * 1e9)}${ext}`;

  const storagePath = options.storagePath || `uploads/${filename}`;
  const firebaseFile = bucket.file(storagePath);

  await firebaseFile.save(file.buffer, {
    metadata: {
      contentType: file.mimetype || "image/jpeg",
    },
    resumable: false,
  });

  const [signedUrl] = await firebaseFile.getSignedUrl({
    action: "read",
    expires: Date.now() + 7 * 24 * 60 * 60 * 1000,
  });

  return { storagePath, url: signedUrl };
}

async function getEntriesForUser({
  userId,
  limit = 50,
  signedUrlTtlMs = 60 * 60 * 1000,
}) {
  const snapshot = await db
    .collection("entries")
    .where("userId", "==", userId)
    .orderBy("createdAt", "desc")
    .limit(limit)
    .get();

  const entries = await Promise.all(
    snapshot.docs.map(async (doc) => {
      const data = doc.data();

      if (!data.storagePath) return null;

      const file = bucket.file(data.storagePath);
      const [url] = await file.getSignedUrl({
        action: "read",
        expires: Date.now() + signedUrlTtlMs,
      });

      return {
        id: doc.id,
        userId: data.userId,
        title: data.title || "",
        description: data.description || "",
        createdAt: data.createdAt,
        storagePath: data.storagePath,
        imageUrl: url,
      };
    }),
  );

  return entries.filter(Boolean);
}

module.exports = { uploadImageToFirebase, createEntry, getEntriesForUser };
