# DermAssistBackend

Get the .env file and serviceAccountKey.json from the discord server and place it in the root directory of the project.

## Installation

npm install

## Running the app

npm run dev

## Consent Form Upload

To upload the consent forms to the database, use the following command:  

node scripts/uploadLatestConsent.js

## ENDPOINTS DOCS

| Endpoint | Method | Params | Output |
|---|---|---|---|
| `/uploads/skin-check` | `POST` | **multipart/form-data**<br>`image` (file, required) | Returns `{ message, requestId, analysis }`.<br><br>`analysis` is a normalized skin-check result:<br>`status`<br>`result.label` = `SKIN` or `NON_SKIN`<br>`result.confidence`<br>`result.predicted_index`<br>`result.attributes`<br>`meta.model` = `Gatekeeper`<br>`meta.latency_ms`<br>`meta.warnings` |
| `/uploads/image` | `POST` | **multipart/form-data**<br>`image` (file, required)<br>`userId` (string, optional, default `demo-user`)<br>`title` (string, optional)<br>`description` (string, optional)<br>`bodyLocation` (string, optional)<br>`tempDelta` (number, optional)<br>`bypassSkinCheck` (boolean-like, optional; accepts `true`, `"true"`, `1`, `"1"`) | Returns `{ message, entry, firebase, analysis, analysisError, bypassSkinCheck, retriedLesionAnalysis }`.<br><br>`entry` = created DB entry<br>`firebase` = upload result / storage metadata<br>`analysis` = normalized lesion-analysis result or `null`<br>`analysisError` = AI failure info or `null`<br>`bypassSkinCheck` = parsed boolean<br>`retriedLesionAnalysis` = `true` if forced retry happened after a `REJECTED_NOT_SKIN` response while bypassing |
| `/uploads/images` | `GET` | **query params**<br>`userId` (string, optional, default `demo-user`)<br>`limit` (number, optional, default `50`, max `200`) | Returns `{ entries }` where `entries` is the list of uploaded entries for the given user |
