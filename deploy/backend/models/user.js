const { db, admin, bucket } = require("../config/firebase");

async function deleteQueryInBatches(query, batchSize = 450) {
  // so we don't go over the 500 firestore limit of documents we can delete in a batch
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 500) {
    throw new Error("batchSize must be an integer between 1 and 500");
  }

  while (true) {
    const snapshot = await query.limit(batchSize).get();
    if (snapshot.empty) break;

    const batch = db.batch();
    snapshot.docs.forEach((doc) => batch.delete(doc.ref));
    await batch.commit();
  }
}

/**
 * Save user profile to Firestore along with the consent data with latest under the user and a log under its collection user.consents
 * @param {string} uid - Firebase Auth UID
 * @param {string} resolvedEmail - User's email address
 * @param {string} fname - User's first name
 * @param {string} lname - User's last name
 * @param {string} dob - User's date of birth
 * @param {Object} consentPayload - Consent data
 */
async function saveUserProfileWithConsent(
  uid,
  resolvedEmail,
  fname,
  lname,
  dob,
  consentPayload,
) {
  const userProfile = await _saveUserProfile(uid, {
    email: resolvedEmail,
    fname,
    lname,
    dob,
  });

  await appendConsentLog(uid, consentPayload);
  return userProfile;
}

/**
 * Save user profile to Firestore
 * @param {string} userId - Firebase Auth UID
 * @param {Object} userData - User profile data
 */
async function _saveUserProfile(userId, userData) {
  try {
    const userRef = db.collection("users").doc(userId);
    const existing = await userRef.get();

    const profile = {
      userId,
      email: userData.email,
      fname: userData.fname,
      lname: userData.lname,
      ...(userData.dob && { dob: userData.dob }),
    };

    if (!existing.exists) {
      profile.createdAt = admin.firestore.FieldValue.serverTimestamp();
    }

    await userRef.set(profile, { merge: true });

    const saved = await userRef.get();
    return { id: saved.id, ...saved.data() };
  } catch (error) {
    console.error("Error saving user profile:", error);
    throw error;
  }
}

/**
 * Get user profile from Firestore
 * @param {string} userId - Firebase Auth UID
 */
async function getUserProfile(userId) {
  try {
    const doc = await db.collection("users").doc(userId).get();
    if (!doc.exists) {
      return null;
    }
    return { id: doc.id, ...doc.data() };
  } catch (error) {
    console.error("Error getting user profile:", error);
    throw error;
  }
}

/**
 * Store user consent in subcollection as we might want to track multiple consents over time
 * Also updates the user profile with the latest consent
 * @param {string} userId
 * @param {Object} consentData
 */
async function appendConsentLog(userId, consentData) {
  try {
    // get the user document
    const userRef = db.collection("users").doc(userId);

    // create a new consent document to be added under the user's consents collection
    const agreedAt = consentData.agreedAt
      ? new Date(consentData.agreedAt)
      : admin.firestore.FieldValue.serverTimestamp();

    const consentRef = userRef.collection("consents").doc();
    const consentDoc = {
      consentType: consentData.consentType,
      version: consentData.version,
      agreedAt,
      ...(consentData.language && { language: consentData.language }),
    };
    // add the new consent to the user's consents subcollection
    await consentRef.set(consentDoc);
    // update the user's profile with the new consent
    await userRef.set({ consent: consentDoc }, { merge: true });

    return consentDoc;
  } catch (error) {
    console.error("Error storing consent:", error);
    throw error;
  }
}

/**
 * Delete user profile and subcollections from Firestore
 * @param {string} userId - Firebase Auth UID
 */
async function deleteUserProfile(userId) {
  try {
    const userRef = db.collection("users").doc(userId);

    // We need to delete the subcollections first before deleting the user document
    // 1. Delete consents subcollection first
    const consentsQuery = userRef.collection("consents");
    await deleteQueryInBatches(consentsQuery);

    // 2. Delete user entries + uploaded files
    // 2.1 Delete the user's folder in the storage bucket (contains uploaded files)
    await bucket.deleteFiles({ prefix: `uploads/${userId}/`, force: true });

    // 2.2 Delete the entries collection for that user
    await deleteQueryInBatches(userRef.collection("entries"));

    // TODO: Delete scans subcollection (to be implemented instead of separate entries collection)

    // 2.3 Delete the user document itself
    await userRef.delete();

    return { success: true };
  } catch (error) {
    console.error("Error deleting user profile:", error);
    throw error;
  }
}

module.exports = {
  saveUserProfileWithConsent,
  appendConsentLog,
  getUserProfile,
  deleteUserProfile,
};
