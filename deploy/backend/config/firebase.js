const admin = require("firebase-admin");
const path = require("path");
const fs = require("fs");
require("dotenv").config();

let serviceAccount;

try {
  const credentialsPathEnv =
    process.env.GOOGLE_APPLICATION_CREDENTIALS ||
    process.env.FIREBASE_SERVICE_ACCOUNT_PATH;

  if (credentialsPathEnv) {
    const credentialsPath = path.isAbsolute(credentialsPathEnv)
      ? credentialsPathEnv
      : path.resolve(process.cwd(), credentialsPathEnv);

    const raw = fs.readFileSync(credentialsPath, "utf8");
    serviceAccount = JSON.parse(raw);
  } else if (process.env.FIREBASE_PROJECT_ID) {
    serviceAccount = {
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, "\n"),
    };
  }
} catch (error) {
  console.warn("Firebase credentials not found or invalid:", error.message);
}

if (serviceAccount && !admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
  });
} else if (!admin.apps.length) {
  admin.initializeApp({
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
  });
}

const db = admin.firestore();
const auth = admin.auth();
const bucket = admin.storage().bucket();

module.exports = { admin, db, auth, bucket };
