const express = require("express");
const router = express.Router();
const authController = require("../controllers/authController");
const consentController = require("../controllers/consentController");

router.post("/signup", authController.signup);
router.post("/login", authController.login);
router.post("/update-consent", authController.updateConsent);
router.delete("/delete-account", authController.deleteAccount);
router.get("/consent-form", consentController.getConsentForm);

module.exports = router;
