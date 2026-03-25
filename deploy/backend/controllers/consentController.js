const { db } = require("../config/firebase");

function toDocId(consentType, version) {
    const normalizedVersion = String(version || "").replace(/_/g, ".");
    const safeVersion = normalizedVersion.replace(/\./g, "_");
    return `${consentType}__${safeVersion}`;
}

function parseVersion(version) {
    return String(version || "")
        .split(".")
        .map((part) => {
            const num = Number(part);
            return Number.isNaN(num) ? 0 : num;
        });
}

function compareVersions(a, b) {
    const pa = parseVersion(a);
    const pb = parseVersion(b);
    const max = Math.max(pa.length, pb.length);
    for (let i = 0; i < max; i += 1) {
        const va = pa[i] || 0;
        const vb = pb[i] || 0;
        if (va > vb) return 1;
        if (va < vb) return -1;
    }
    return 0;
}

// Gets the latest version of the consent form
// Good to keep on backend since we search through all consent forms to find the latest version
const getConsentForm = async (req, res) => {
    const { consentType, version, locale } = req.query;

    if (!consentType) {
        return res.status(400).json({ error: "consentType is required" });
    }

    try {
        let consentDoc = null;

        if (version) {
            const docId = toDocId(consentType, version);
            const docSnap = await db.collection("consent_forms").doc(docId).get();
            if (!docSnap.exists) {
                return res.status(404).json({ error: "Consent form not found" });
            }
            consentDoc = docSnap.data();
        } else {
            const snapshot = await db.collection("consent_forms")
                .where("consentType", "==", consentType)
                .where("status", "==", "published")
                .get();

            if (snapshot.empty) {
                return res.status(404).json({ error: "Consent form not found" });
            }

            // Find the latest version
            snapshot.forEach((docSnap) => {
                const data = docSnap.data();
                if (!consentDoc || compareVersions(data.version, consentDoc.version) > 0) {
                    consentDoc = data;
                }
            });
        }

        if (!consentDoc) {
            return res.status(404).json({ error: "Consent form not found" });
        }

        // Return localized content if locale is specified
        if (locale) {
            const localized = consentDoc.content ? consentDoc.content[locale] : null;
            if (!localized) {
                return res.status(404).json({ error: "Locale not found for consent form" });
            }
            return res.status(200).json({
                ...consentDoc,
                content: localized,
                locale,
            });
        }

        return res.status(200).json(consentDoc);
    } catch (error) {
        console.error("Get consent form error:", error);
        return res.status(500).json({ error: error.message || "Failed to fetch consent form" });
    }
};

module.exports = {
    getConsentForm,
};
