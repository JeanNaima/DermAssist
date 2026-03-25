const { db, bucket, admin } = require("../config/firebase");

function getUserRef(userId) {
  return db.collection("users").doc(userId);
}

function getEntryRef(userId, entryId) {
  return getUserRef(userId).collection("entries").doc(entryId);
}

function createEntryRef(userId) {
  return getUserRef(userId).collection("entries").doc();
}

async function uploadImageToFirebase(file, { storagePath, metadata = {} }) {
  const fileRef = bucket.file(storagePath);

  await fileRef.save(file.buffer, {
    resumable: false,
    metadata: {
      contentType: file.mimetype,
      metadata,
    },
  });

  return {
    storagePath,
    bucket: bucket.name,
    contentType: file.mimetype,
    size: file.size,
  };
}

const acronymMap = {
  "nv": "Melanocytic nevi",
  "mel": "Melanoma", 
  "bkl": "Benign keratosis-like lesions", 
  "bcc": "Basal cell carcinoma", 
  "akiec": "Actinic keratoses", 
  "vasc": "Vascular lesions", 
  "df": "Dermatofibroma",  
  "scr": "Scar/Regular skin", 
};

function mapAnalysisToAiResult(analysis) {
  if (!analysis || analysis.status !== "success") return null;

  const attributes = analysis.result?.attributes ?? {};

  const top5Results = [...attributes]
  .sort((a, b) => b.confidence - a.confidence)
  .slice(0, 5);
    
  return {
    confidence: Number(analysis.result?.confidence ?? 0),
    predictedResult: analysis.result?.label ?? "UNKNOWN",
    top5Results,
  };
}

async function createEntry({
  userId,
  entryId,
  storagePath,
  title,
  description,
  bodyLocation,
  tempDelta = null,
  aiResult = null,
}) {
  const ref = entryId ? getEntryRef(userId, entryId) : createEntryRef(userId);

  const entryData = {
    entryId: ref.id,
    userId,
    storagePath,
    title: title || "",
    description: description || "",
    bodyLocation: bodyLocation || "",
    ...(tempDelta !== null && { tempDelta: Number(tempDelta) }),
    ...(aiResult && { aiResult }),
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  };

  await ref.set(entryData, { merge: true });

  const saved = await ref.get();
  return { id: saved.id, ...saved.data() };
}

async function updateEntry(userId, entryId, updates) {
  const ref = getEntryRef(userId, entryId);
  await ref.set(updates, { merge: true });
}

async function getEntriesForUser({ userId, limit = 50 }) {
  const snapshot = await getUserRef(userId)
    .collection("entries")
    .orderBy("createdAt", "desc")
    .limit(limit)
    .get();

  return snapshot.docs.map((doc) => ({
    id: doc.id,
    ...doc.data(),
  }));
}

module.exports = {
  uploadImageToFirebase,
  createEntry,
  updateEntry,
  getEntriesForUser,
  createEntryRef,
  mapAnalysisToAiResult,
  acronymMap,
};