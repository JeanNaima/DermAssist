const { db, admin } = require("../config/firebase");

/**
 * Save user profile to Firestore
 * @param {string} userId - Firebase Auth UID
 * @param {Object} userData - User profile data
 */
async function saveUserProfile(userId, userData) {
    try {
        const userRef = db.collection("users").doc(userId);
        const profile = {
            userId,
            email: userData.email,
            fname: userData.fname,
            lname: userData.lname,
            createdAt: admin.firestore.FieldValue.serverTimestamp(),
            ...(userData.dob && { dob: userData.dob }),
        };
        await userRef.set(profile);
        return profile;
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
 * Store user consent
 * @param {string} userId
 * @param {Object} consentData
 */
async function storeConsent(userId, consentData) {
    try {
        const consentRef = db.collection("users").doc(userId).collection("consents").doc();
        const consentDoc = {
            consentType: consentData.consentType,
            version: consentData.version,
            acceptAt: admin.firestore.FieldValue.serverTimestamp(),
        };
        await consentRef.set(consentDoc);
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

        // 1. Delete consents subcollection first
        const consentsSnapshot = await userRef.collection("consents").get();
        const batch = db.batch();
        consentsSnapshot.docs.forEach((doc) => {
            batch.delete(doc.ref);
        });
        await batch.commit();

        // TODO: Delete scans subcollection

        // 2. Delete the user document itself
        await userRef.delete();

        return { success: true };
    } catch (error) {
        console.error("Error deleting user profile:", error);
        throw error;
    }
}

module.exports = {
    saveUserProfile,
    getUserProfile,
    storeConsent,
    deleteUserProfile,
};