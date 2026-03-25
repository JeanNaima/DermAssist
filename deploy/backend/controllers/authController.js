const { auth } = require("../config/firebase");
const User = require("../models/user");

/**
 * Handle user signup (profile creation only; Auth handled on frontend)
 */
const signup = async (req, res) => {
    console.log("Signup request:", req.body);
    const { uid, fname, lname, email, dob, consent } = req.body;
    const authHeader = req.headers.authorization || "";
    const idToken = authHeader.startsWith("Bearer ")
        ? authHeader.slice(7)
        : null;

    if (!idToken || !fname || !lname || !uid) {
        return res.status(400).json({ error: "ID Token, uid, first name, and last name are required" });
    }

    try {
        // Verify the ID token to get the UID and email
        const decodedToken = await auth.verifyIdToken(idToken);
        const tokenUid = decodedToken.uid;
        if (uid !== tokenUid) {
            return res.status(403).json({ error: "Token UID does not match request UID" });
        }
        const resolvedEmail = email || decodedToken.email;

        if (!resolvedEmail) {
            return res.status(400).json({ error: "Email is required" });
        }

        const consentType = consent?.consentType;
        const consentPayload = consent
            ? {
                consentType,
                version: consent.version,
                language: consent.language,
                agreedAt: consent.agreedAt,
            }
            : null;

        // Store profile in Firestore
        const userProfile = await User.saveUserProfileWithConsent(uid, resolvedEmail, fname, lname, dob, consentPayload);

        res.status(200).json({
            message: "successful",
            user: userProfile,
        });
    } catch (error) {
        console.error("Signup error:", error);
        res.status(500).json({ error: error.message || "Failed to create user profile" });
    }
};

/**
 * Handle user login (Token verification)
 */
const login = async (req, res) => {
    const authHeader = req.headers.authorization || "";
    const idToken = authHeader.startsWith("Bearer ")
        ? authHeader.slice(7)
        : null;

    console.log("Login request");

    try {
        if (!idToken) {
            return res.status(400).json({ error: "ID Token required" });
        }

        // Verify the ID token (secure method)
        const decodedToken = await auth.verifyIdToken(idToken);
        const uid = decodedToken.uid;

        // Get user profile from Firestore
        const userProfile = await User.getUserProfile(uid);

        if (!userProfile) {
            return res.status(404).json({ error: "User profile not found" });
        }

        res.status(200).json({
            message: "successful",
            user: userProfile,
        });
    } catch (error) {
        console.error("Login error:", error);
        res.status(401).json({ error: "Authentication failed" });
    }
};

const updateConsent = async (req, res) => {
    const authHeader = req.headers.authorization || "";
    const idToken = authHeader.startsWith("Bearer ")
        ? authHeader.slice(7)
        : null;

    if (!idToken) {
        return res.status(400).json({ error: "ID Token is required" });
    }

    try {
        // 1. Verify the ID token to get the UID
        const decodedToken = await auth.verifyIdToken(idToken);
        const uid = decodedToken.uid;

        // 2. Update consent in Firestore
        const consentData = req.body;
        await User.appendConsentLog(uid, consentData);

        res.status(200).json({
            message: "Consent updated successfully",
        });
    } catch (error) {
        console.error("Update consent error:", error);
        res.status(500).json({ error: error.message || "Failed to update consent" });
    }
};

/**
 * Handle account deletion
 */
const deleteAccount = async (req, res) => {
    const authHeader = req.headers.authorization || "";
    const idToken = authHeader.startsWith("Bearer ")
        ? authHeader.slice(7)
        : null;

    if (!idToken) {
        return res.status(400).json({ error: "ID Token is required" });
    }

    try {
        // 1. Verify the ID token to get the UID
        const decodedToken = await auth.verifyIdToken(idToken);
        const uid = decodedToken.uid;

        // 2. Delete from Firestore (profile and subcollections)
        await User.deleteUserProfile(uid);

        // 3. Delete from Firebase Auth last
        await auth.deleteUser(uid);

        res.status(200).json({
            message: "Account deleted successfully",
        });
    } catch (error) {
        console.error("Delete account error:", error);
        res.status(500).json({ error: error.message || "Failed to delete account" });
    }
};

module.exports = {
    signup,
    login,
    updateConsent,
    deleteAccount,
};
