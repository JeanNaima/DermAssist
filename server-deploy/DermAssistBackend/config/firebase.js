const admin = require("firebase-admin");
const path = require("path");
require("dotenv").config();

let serviceAccount;

try {
  // If you have a service account file, use it, otherwise use environment variables
  if (process.env.FIREBASE_SERVICE_ACCOUNT_PATH) {
    const serviceAccountPath = path.resolve(
      process.cwd(),
      process.env.FIREBASE_SERVICE_ACCOUNT_PATH,
    );
    serviceAccount = require(serviceAccountPath);
  } else if (process.env.FIREBASE_PROJECT_ID) {
    serviceAccount = {
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, "\n"),
    };
  }
} catch (error) {
  console.warn("Firebase credentials not found. Ensure .env is configured.");
}

if (serviceAccount && !admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
  });
} else if (!admin.apps.length) {
  // Try default credentials (e.g., if running in GCP)
  admin.initializeApp();
}

const db = admin.firestore();
const auth = admin.auth();
const bucket = admin.storage().bucket();

module.exports = { admin, db, auth, bucket };
