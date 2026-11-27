# AI security

What could go wrong with To AI or Not to AI, how likely it is, what I've done about it, and what's still open.

I wrote this to practise the way the ISO standards think about risk, on a real project instead of a textbook example. The process comes from **ISO/IEC 23894** (AI risk management, built on ISO 31000): understand the context, find the risks, score them, decide what to do, and be honest about what's left. The controls are mapped to **ISO/IEC 27001:2022 Annex A** for general security and to **ISO/IEC 42001** for the AI-specific parts.

> This is practice, not a certified management system. I've mapped controls as carefully as I can, but check the numbers against the published standards before quoting them anywhere. 42001 is mapped only at the section level.

---

## What we're protecting

To AI or Not to AI is a website, a browser extension and an API that guess whether an image was made by AI. A small model (MobileNetV3-Small) gives one of five verdicts: human made, AI generated, or one of three shades of "honestly not sure". Each comes with a confidence score and, if you want, a heatmap of where it looked. People can vote on whether it got it right.

It runs on free tiers: the API on Render (one small instance that naps when idle), the database on Neon, Redis on Upstash. Training happens by hand on Kaggle, from a fixed commit of this repo.

The things worth protecting:

| Asset | Why it matters |
|---|---|
| The model and its thresholds | Tamper with these and every answer is wrong |
| The real API key | It opens the admin routes |
| Prediction logs and votes | They drive drift monitoring, and filenames can contain personal info |
| Uploaded images | People's pictures; they should never be kept |
| Staying online | One free instance is easy to knock over |

Who's affected: the people using the site and the extension, me as the one running it, the people *in* the pictures, and anyone a verdict gets used against.

Not covered here: the security of Render, Neon, Upstash, Kaggle and GitHub themselves. That's their job (27001 control 5.23 is the one about trusting your cloud providers).

---

## How risks are scored

Each risk gets a likelihood and an impact from 1 to 3. Multiply them to get the level.

| Score | Likelihood | Impact |
|---|---|---|
| **1** | Needs real skill, access or luck | Annoying, nothing lasting |
| **2** | Plausible for someone motivated | Wrong answers or downtime for many people, or personal data exposed |
| **3** | Easy, or already happening | Misleading results with real consequences, or admin access lost |

What the level means:

- **1–2, low:** fine. Write it down and move on.
- **3–4, medium:** fix it if the fix is cheap; otherwise keep an eye on it.
- **6–9, high:** fix it, or accept it out loud with a reason.

---

## The register

| ID | Risk | L | I | Level | What I'm doing | Left over |
|---|---|---|---|---|---|---|
| R1 | Fooling the model with everyday edits (re-save, resize, blur, crop) | 3 | 2 | **6 High** | Reduce, then accept | Medium |
| R2 | Adversarial pixels crafted against the model | 2 | 2 | 4 | Accept | Medium |
| R3 | Fake votes triggering retraining | 2 | 1 | 2 | Reduce | Low |
| R4 | API abuse, costs, copying the model | 2 | 1 | 2 | Accept | Low |
| R5 | Huge or malicious uploads taking the server down | 2 | 2 | 4 | Reduce | Low |
| R6 | Personal info in logs, kept forever | 2 | 2 | 4 | Reduce | Low |
| R7 | Leaked secrets or someone getting admin access | 1 | 3 | 3 | Reduce | Low |
| R8 | A CORS policy that's wider than it needs to be | 1 | 1 | 1 | Reduced | Low |
| R9 | A tampered model file or a vulnerable dependency | 1 | 3 | 3 | Reduce | Low |
| R10 | The model going stale as new generators appear | 3 | 2 | **6 High** | Reduce, then accept | Medium |
| R11 | Someone treating a verdict as proof | 2 | 3 | **6 High** | Reduce, then accept | Medium |

---

## The risks, one by one

### R1. Fooling the model with everyday edits

This is the easy attack. Take an AI image, re-save it, shrink it, blur it a little, and see if the verdict flips. No skill required.

**What's in place.** Every training image gets JPEG-compressed at a random quality, so file format and compression can't give the game away. Each release is tested against JPEG quality 75, half-size resizing and a slight blur, and those scores sit right next to the clean one. And when the model isn't sure, it says so, instead of guessing confidently.

**What's left.** I deliberately didn't train on blur or heavy resizing, because they wipe out the tiny traces that make detection possible in the first place. So they're still the cheapest way to fool it. I'm accepting that, saying so here, and leaving the better fixes for v3.

*Maps to:* 27001 8.29 (security testing before release) · 42001 A.6 (testing across the AI life cycle)

### R2. Adversarial pixels

The sophisticated version of R1: invisible pixel changes calculated specifically to fool this model. The model files are public in this repo, so an attacker has everything they need. The heatmaps even show where the model looks.

**What's in place.** Nothing specific, and that's a choice.

**What's left.** I'd rather the project be transparent (open model, visible heatmaps) than secure through obscurity. Defending against adversarial attacks properly is a research problem, not a weekend fix. Accepted and stated openly.

*Maps to:* 42001 A.6 (robustness) and A.8 (being upfront about limits)

### R3. Fake votes triggering retraining

Votes are anonymous, so someone could flood them to make the model look broken.

**What's in place.** The drift check is hard to push around: it needs a minimum number of votes, only looks at recent ones, ignores "I don't know" answers and the model's own "not sure" verdicts, and only counts votes about the model that's live right now. Only one retraining job can be queued at a time, and the demo key's votes are rate-limited per IP.

The most important part: **votes never become training data.** Retraining uses the curated dataset; votes only decide *whether* a retraining job gets queued. And admin labels need the real key and are stored separately.

**Something I found while writing this.** The automatic retraining job can't actually succeed in production. The worker has no GPU and no dataset, and even if it trained a model, the API wouldn't see it, because it lives in a different container. In reality, retraining is manual: a Kaggle run, me reading the results, then a pull request. That manual review is the real safety net, and it's a good one. But a job that quietly fails and retries gives the wrong impression, so it should notify me instead (action A5).

*Maps to:* 27001 8.16 (monitoring) and 8.32 (change management) · 42001 A.6 (life cycle) and A.7 (data)

### R4. API abuse, costs, copying the model

Someone scripting the public demo to death, or querying it to clone the model.

**What's in place.** Two kinds of key: the public demo key only reaches analysis and feedback, while admin routes need the real one (and there are tests proving the demo key gets a 401). The demo has per-IP rate limits, and every route has a global limit. Answers are a label and a rounded confidence, not raw model outputs.

**What's left.** Honestly, not much. Copying the model through the API would be pointless, since it's published right here. And on a free tier, the worst case is a slow afternoon, not a bill. Accepted.

*Maps to:* 27001 5.15 (access control), 8.5 (secure authentication) and 8.6 (capacity management)

### R5. Huge or malicious uploads

Files built to eat all the memory or crash the image decoder. On a single small instance, that means downtime.

**What's in place.**
- **A 10 MB limit.** Until recently it only checked the `Content-Length` header, so an upload that left the header out could be any size and would still be read into memory in full. That's fixed now: the read itself stops at the limit.
- **File type checked from the actual bytes,** not the file name, and only JPEG and PNG are accepted.
- **More checks:** a minimum image size, Pillow's protection against decompression bombs, and cleaned-up filenames.
- **Shrinking in the browser:** the website resizes anything big before uploading it at all.

*Maps to:* 27001 8.6 (capacity), 8.26 (application security requirements) and 8.28 (secure coding)

### R6. Personal info in logs

Images are never stored. I checked the code: they're analysed in memory and thrown away. But filenames *are* logged, and people name files things like `ayse_passport.jpg`. Also, logs and votes are currently kept forever.

**What's in place.** No image bytes on disk, in the database or in the cache. The cache keeps only the verdict, keyed by a fingerprint of the image, for 24 hours. The privacy page says what's kept, and contact goes through GitHub issues, so no personal email is public. The extension can't see any website until you allow it.

**What's left.** Log a shortened or hashed filename instead of the real one, and delete logs older than a set period, say 90 days (action A4).

*Maps to:* 27001 5.34 (privacy and personal data), 8.10 (deleting information) and 8.11 (data masking) · 42001 A.7 (data)

### R7. Leaked secrets or admin access

**What's in place.** Secrets live only in environment variables: Render's settings in production, and a git-ignored `.env` locally. The real key was rotated after it got exposed once during development; that's how I learned this one. The demo key is public on purpose and can't open admin routes. The container doesn't run as root.

**What's left.** Turn on GitHub's secret scanning, so a key committed by accident gets caught right away (action A6).

*Maps to:* 27001 5.17 (authentication information), 8.2 (privileged access) and 8.5 (secure authentication)

### R8. CORS wider than needed

During development, the allowed origins included `"*"`, meaning any website could call the API from a visitor's browser. Because the API uses a key in a header rather than cookies, that was never a way to act as a logged-in user, so the real risk was tiny. It's now locked to the site itself: the website is served by the API (same origin, no CORS needed), and the extension has its own permission for the API's address, so nothing legitimate needed the wildcard.
*Maps to:* 27001 8.9 (configuration management) and 8.26 (application security requirements)

### R9. A tampered model or a vulnerable dependency

**What's in place.** Every dependency is pinned to an exact version. Each model records the exact commit that trained it, the training notebook writes checksums for the model files, and the pretrained starting weights come from PyTorch's official source.

**What's left.** The API doesn't check the model's checksum when it starts, so a swapped file would go unnoticed (action A2). And nothing automatically checks dependencies for known vulnerabilities (action A6).

*Maps to:* 27001 5.21 (supply chain), 8.8 (handling known vulnerabilities) and 8.32 (change management) · 42001 A.10 (third parties)

### R10. The model going stale

New image generators come out all the time, and a detector only knows what it's been shown. This one is guaranteed to happen; the only question is how fast.

**What's in place.** Two datasets are kept completely out of training, and the model's score on them is reported separately. That number is the honest preview of how it'll do on fakes it's never seen. Every answer says which model version gave it, votes feed drift monitoring, and the uncertainty zone catches some of what it doesn't know.

**What's left.** Retrain with newer generators each release, always keeping one generator out of training as a test (action A8). Until then, it's a stated limitation.

*Maps to:* 42001 A.5 (impact assessment), A.6 (operation and monitoring) and A.8 (being upfront with users)

### R11. A verdict treated as proof

This is the one I care about most, and it isn't technical at all. Someone runs a photo through the site, gets "AI generated, 87% sure", and uses it to accuse a person of faking it. Or the other way around: a real fake gets waved through as "human made".

**What's in place.** That's exactly why there are five verdicts instead of two, a confidence on every answer, and a heatmap that shows what the model looked at. The README and model card spell out its limits.

**What's left.** A tool can't control how people use it, but it can be clear about what it is. Every verdict should say plainly that it's a probability, not proof, on the website and in the extension (action A7).

*Maps to:* 42001 A.5 (impact assessment), A.8 (information for users) and A.9 (responsible use)

---

## What's next

| # | Risk | Action | Status |
|---|---|---|---|
| A1 | R5 | Stop reading uploads at the size limit, not just check the header | **Done, waiting to merge** |
| A2 | R9 | Check the model's checksum at startup and refuse to run if it doesn't match | To do |
| A3 | R8 | Replace the CORS wildcard with the real site only | **Done** |
| A4 | R6 | Log shortened or hashed filenames; delete logs and votes older than 90 days | To do |
| A5 | R3 | Make the drift job notify me instead of attempting training; document how retraining really works | To do |
| A6 | R7, R9 | Turn on GitHub secret scanning, push protection and Dependabot | **Done** |
| A7 | R11 | Show "a probability, not proof" with every verdict | **Done** |
| A8 | R10 | Retrain with newer generators each release, keeping one held out | Ongoing |

Every risk and action here is owned by me.

---

## Keeping this honest

A risk register that never changes is just a document. This one gets revisited:

- whenever a new model ships, because new numbers can change R1, R10 and R11;
- whenever the project starts accepting a new kind of input or talks to a new service;
- at least every three months either way.

| Date | What changed |
|---|---|
| 2026-11-24 | First version |