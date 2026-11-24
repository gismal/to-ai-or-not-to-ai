# Engineering notes

How AI or Not AI is built, why it's built that way, and what went wrong along the way.

The first half is the decisions: what I chose, why, and what I gave up for it. Every choice here had a cost, and I'd rather name it than pretend it didn't. The second half is the problems: the bugs and dead ends, and what each one taught me.

---

## Design decisions

### The model

#### Five verdicts instead of two

Most detectors answer "AI" or "not AI" for every image, even when they're guessing. This one has an honest middle: besides **Human made** and **AI generated**, it can say **Uncertain**, leaning one way, leaning the other, or genuinely torn.

The thresholds aren't picked by feel. They're chosen on the validation set so that when the model *does* commit to an answer, it's right at least 95% of the time. Alongside accuracy, I report how often the model decides at all. "95% accurate on the 70% of images it's willing to judge" is a less impressive sentence than "92% accurate", but it's the true one, and it's what a user actually experiences.

**What I gave up:** a simpler headline number, and some answers. A model that sometimes says "I don't know" looks weaker on a leaderboard, even when it's more useful in real life.

#### A small model: MobileNetV3-Small

The model runs on a free CPU server that sleeps when idle, and it has to answer quickly once it wakes up. MobileNetV3-Small is about 6 MB, loads in a moment, and answers in tens of milliseconds on a CPU.

**What I gave up:** accuracy headroom. A bigger model (a ViT, an EfficientNet-B3) would likely score higher, and would also be slower, more expensive to host, and harder to retrain on free GPUs.

#### Two-phase training

Training starts from ImageNet weights. Phase one freezes the backbone and trains only the classifier head (about 590,000 parameters), so the new head learns its job without wrecking features it hasn't understood yet. Phase two unfreezes everything and fine-tunes at a learning rate ten times lower. The best weights across *both* phases win, so a fine-tuning phase that makes things worse can't replace a better head-only model.

**What I gave up:** a simpler training loop. Two phases mean two optimizers, two early-stopping counters, and more settings to log.

#### JPEG re-compression instead of blur or rotation

The face data had a trap: 98% of the AI images were PNG, while 71% of the real photos were JPEG. A model can learn "PNG means AI" and score well without learning anything about faces. So every training image goes through JPEG compression at a random quality, which makes file format useless as a clue.

I deliberately skipped blur and rotation. Blur smooths away the fine pixel-level traces that generators leave, which are exactly the signal I want the model to find. Rotation doesn't match how people share images.

**What I gave up:** some robustness to blurry or rotated uploads, in exchange for keeping the fingerprints the detection depends on.

#### ONNX for serving, PyTorch for explaining

The API runs the model through ONNX Runtime: no PyTorch in the request path, a smaller image, faster startup. The PyTorch weights are kept only for GradCAM, the heatmap that shows where the model looked.

**What I gave up:** having one artifact. There are two model files that must always come from the same training run, so the export step checks that the ONNX output matches PyTorch before anything ships.

### The data

#### No single dataset gets to dominate

The training data comes from about a dozen public datasets of very different sizes. Images are picked in rotation, one per source in turn, so a huge dataset can't drown out a small one, and the model can't learn the "style" of one dataset instead of the difference between real and generated. Tiny images are dropped, and the rules that decide each folder's label are printed so a human can check them.

**What I gave up:** raw volume. The capped, balanced set is smaller than everything available combined.

#### Some fakes the model never sees in training

Two datasets are held out completely: Photoshop-edited faces (RealFakeFace) and deepfake video frames (FaceForensics). They appear only in testing, as a check on how the model handles kinds of manipulation it was never shown. That number is reported separately from the main score.

**What I gave up:** training data, and a flattering single score.

#### Near-duplicates are removed from val and test

Video-based datasets are full of nearly identical frames. A perceptual hash finds validation and test images that are near-copies of something earlier in the split, and removes them, so the test score measures generalization rather than memory.

#### The training code lives in the repo, not the notebook

Kaggle provides the GPU, but the notebook only orchestrates: it clones the repository at a fixed commit and runs `scripts/train.py`. The commit is printed and saved with the model's metadata, so any model can be traced back to the exact code that trained it.

**What I gave up:** quick experiments in the notebook. Every training change has to be committed first.

### Serving

#### Every answer says which model gave it

The model's version and training date are read from its metadata file at startup, reported by `/health`, and included in every prediction. Feedback can then be tied to the model it was about, so the drift check ignores votes about older models.

#### The cache is keyed by what the image looks like and which model saw it

Results are cached by a perceptual hash of the image, so a visually identical image (re-saved, slightly recompressed) is a cache hit. The model version is part of the key, so deploying a new model never serves the old model's answers. The cache fails open: if Redis is down or a read fails, the image is simply analysed again.

**What I gave up:** some hits. A new model starts with a cold cache.

#### Two kinds of key, and a public demo

There are two keys. The real key opens everything, including admin routes. The demo key is public on purpose (it's in the page's source) and only opens the analysis and feedback routes, with a per-IP rate limit on each. Anyone can try the site without signing up, and the demo can't be used for admin access or unlimited traffic.

**What I gave up:** per-user accounts and per-client limits. A table for per-client keys existed early on and was removed, unused. It's on the future-work list instead.

#### Free-tier hosting, with the cold start shown instead of hidden

The API runs on Render's free tier, the database on Neon, and Redis on Upstash. It costs nothing, but Render puts the server to sleep after about 15 minutes idle, and waking takes up to a minute.

Instead of hiding that, the page pings `/health` as soon as it loads. If the server is asleep, a theatre curtain covers the upload area until it's awake, so most of the wait happens while the visitor is still choosing an image. An analysis that takes longer than 8 seconds brings the curtain down too.

**What I gave up:** a fast first visit. On a paid instance this whole problem disappears.

#### Migrations run when the container starts

The Dockerfile runs `alembic upgrade head` before starting the server, so a deploy can't start with a database that's behind the code.

**What I gave up:** a little startup time, and the option to run several instances at once without them racing to migrate. With one instance, that's fine.

#### Logging never blocks or breaks an answer

Each prediction is logged in a background task with its own database session, after the response has been sent. If logging fails, the user never knows.

### The feedback loop

#### Every vote is kept

People can vote on whether a verdict was right, as often as they like. Every vote is stored; when one is needed per image, the latest one counts. Each vote is classified when it's saved (false positive, false negative, uncertain miss, correct, or unknown), and that classification is a required argument, not something the database guesses later.

**What I gave up:** convenience at the call sites, which must always pass the classification explicitly.

#### The drift check is hard to fool

The drift check measures the error rate over recent feedback, joined to the actual logged predictions, within a time window. It ignores "I don't know" answers and the model's own abstentions, requires a minimum number of votes before it can trigger anything, and queues at most one retraining job at a time.

**What I gave up:** speed of reaction. A real drop in quality takes a while to show up, but a handful of angry clicks can't trigger retraining.

### The clients

#### One HTML file, no build step

The website is a single HTML file: React and Babel from a CDN, styles inline, no bundler. It's trivial to deploy and to read.

**What I gave up:** a faster first load (Babel compiles in the browser), type checking, and component files. For one page, that trade is worth it. For a bigger app, it wouldn't be.

#### The extension asks for access only when it needs it

A fresh install can reach only the API. Analysing an image on a site asks for access to that one site; turning on passive mode asks for all sites. Users and store reviewers both trust an extension more when it asks at the moment it needs something.

**What I gave up:** a few extra clicks the first time on each site.

---

## Problems we faced

Every project has a version of its history where everything went to plan. This isn't that version. Each problem below comes with what broke, why, how I found it, and what I'd tell my past self.

### Production: where "works on my machine" goes to die

#### The 502 that only existed on Render

`/explain` worked perfectly on my laptop and failed every single time in production. Classic.

The culprit was a background task. After answering a request, the API logged the prediction to the database, but it used the request's own database session, which was already closed by then. That killed the connection, and Render's proxy then happily reused the dead connection for the next request.

The fix: the logging task now opens its own session and swallows its own errors. Logging a prediction should never be able to break the prediction.

**Lesson:** background tasks outlive the request that started them. Give them their own stuff.

#### The table that never existed

Render's logs told me `prediction_logs` didn't exist. Which was news to me, because it existed just fine locally.

It turned out both of my Alembic migrations were empty. Locally the tables had been created some other way, so nothing ever looked wrong. Production, starting from a clean database, simply never got them.

I replaced the empty migrations with one proper baseline that creates all five tables, and the Dockerfile now runs `alembic upgrade head` before the server starts. (It was also running a second time inside the app's startup. It doesn't anymore.)

**Lesson:** a model describes a table, a migration actually builds it. A working local database proves nothing about a fresh one.

#### Three deploys, three different ways to fail

Fixing the migrations broke the next deploy: Neon's `alembic_version` table still pointed at one of the migrations I'd just deleted. Dropping that table and redeploying fixed it.

Then, in the middle of all this, the database URL on Render got changed to plain `postgresql://`, the format `psql` uses. SQLAlchemy read that as "use psycopg2", which isn't installed, and the deploy fell over again. The app needs `postgresql+asyncpg://`. The driver is part of the URL, and the URL your terminal likes isn't the one your app likes.

And then Upstash, my hosted Redis, refused to connect at all: "Only 0th database is supported." My code had hard-coded database 1 for the cache, because locally you get sixteen databases for free. Upstash gives you one. The code now uses the URL exactly as given.

**Lesson:** managed services quietly remove assumptions you didn't know you were making. Rewriting history (like migrations) means every database that remembers the old history has to be told.

#### Votes that couldn't be cast twice

Users couldn't vote on the same image twice, and some clients couldn't vote at all. Two separate problems were hiding behind one symptom:

- A unique constraint allowed exactly one vote per filename and source. Fine for a first draft, wrong for real use.
- The list of allowed clients didn't include the web playground or the browser extension, which happen to be the two places people actually vote from.

Dropping the constraint needed a hand-written migration, my first one, with `DROP CONSTRAINT IF EXISTS` because not every database had the constraint in the first place.

**Lesson:** autogenerate can't read your mind; writing a migration by hand is completely normal. And `IF EXISTS` saves you when your environments have quietly drifted apart.

#### A rate limit that limited nothing

The demo key was supposed to be rate-limited. The setting existed, the limiter existed, and neither was attached to a single route. The smoke test's rate-limit check against production is what caught it.

**Lesson:** configuration isn't behaviour. If something matters, test it from the outside.

#### One comma, one 500

The admin "uncertain predictions" page crashed even with the right key. The cause was a single trailing comma after `.limit(limit),`, which quietly turned the query into a tuple. A `# type: ignore` on that line had been hiding the warning that would have pointed straight at it.

**Lesson:** `# type: ignore` is a promise that you know better than the type checker. I didn't.

#### The test that failed only on slow machines

The rate-limit test fired six requests and expected the sixth to be blocked within one second. On my machine it was. On a busy CI runner it sometimes wasn't. The limit is now a setting, and the test uses a per-minute limit that no runner is slow enough to miss.

**Lesson:** a test that depends on how fast your computer is will eventually meet a slower computer.

---

### The cache that never cached (my favourite one)

This one deserves its own section, because it was invisible and a passing test was actively covering for it.

The API caches results, so the same image doesn't get analysed twice. Except it always did. Every cache read failed because of `json.load(raw)` where it should have been `json.loads(raw)`. One letter. The error was caught and logged as a warning, so nothing ever looked broken. It was just slower than it should have been, forever.

When I fixed that letter, the code behind it ran for the first time ever, and it crashed immediately (a duplicate argument). It also had a bug of its own: on a cache hit it returned the *first* upload's filename, so uploading the same picture as `holiday.jpg` would show the name of whatever someone had uploaded before.

The worst part: there was a test called `test_cache_hit_on_second_request`, and it passed the whole time. It mocked the entire inference service, so it never touched the cache at all. It only proved that the route faithfully passes along whatever the mock says.

The new test runs the real service and the real cache against a tiny fake Redis that stores strings, the way real Redis does. It would have caught all three bugs.

**Lesson:** a test proves only what it doesn't mock. And an exception that gets caught and logged is still a failure, just a quiet one.

---

### Training data: where the model tries to cheat

#### PNG = AI, apparently

Before training, I checked what file formats each class used. In the face data, 98% of AI images were PNG, while 71% of real photos were JPEG. A model doesn't care *why* the classes differ; if "is it a PNG?" gets the right answer most of the time, it will happily learn that instead of anything about faces.

The fix is JPEG re-compression on every training image, so file format carries no signal at all.

**Lesson:** before training, check what's trivially different between your classes. If you don't find the shortcut, the model will.

#### Twins in the test set

One dataset is built from video frames, and neighbouring frames are nearly identical. A random split puts one frame in training and its twin in testing, and suddenly the test score is measuring memory, not skill. Perceptual hashing found and removed the twins: 194 from validation and 799 from test.

I also held two datasets out of training completely (Photoshop-edited faces and deepfakes). They're used only for testing, to see how the model handles kinds of fakes it has never seen. That number is less flattering than the main score, and more honest.

**Lesson:** a random split isn't an independent split when your data has near-duplicates.

---

### Kaggle: the long nights

Training runs on Kaggle's free GPUs, and this is where I lost the most hours.

**Huge images.** A handful of images were up to 161 megapixels, and the hashing step decoded each one at full size, eight at a time. The run slowed to a crawl and the cell sat there showing `*`. Images over 50 MP are now skipped, and hashing works on a reduced-size copy.

**Slow, or stuck?** Picking images opened files one at a time on Kaggle's slow input disk. Worse, if one dataset was mostly too-small images, the code read nearly all of it before moving on. One step took over 85 minutes and printed nothing, so it looked exactly like a frozen notebook. Now it checks files in parallel, gives up on a source after 300 unusable files in a row, and shows progress.

**The truncated image.** After two hours of preparing data, training crashed two minutes in, on one corrupted file. It had passed the size check because that only reads the file's header. The hashing step *had* failed on it, but rows without a hash were kept anyway. Now anything that can't be fully decoded is dropped before training.

**The sleeping laptop.** Interactive Kaggle sessions need your browser connected, so closing the laptop ended the run. Committed runs ("Save & Run All") run on Kaggle's servers and don't care what my laptop is doing.

**Lesson:** any long loop needs visible progress, because slow and stuck look identical from the outside. And validate data the same way it'll be used: a header check isn't a decode.

---

### Smaller ones worth a mention

- **The extension asked for access to every website at install.** Now that's optional: it asks for one site when you analyse an image there, and for all sites only if you turn on passive mode. A fresh install can reach nothing but the API. Users trust that more, and so do store reviewers.
- **A new favicon refused to show up.** Browsers cache favicons by URL, so `favicon.png?v=2` fixed it for everyone.
- **ONNX export broke after a PyTorch update.** Exporting at opset 18 instead of 12 fixed it.
- **Drift checks returned `no_data` in tests.** Drift joins feedback to logged predictions, and the test setup created feedback without the matching prediction logs. The tests now seed both.
- **Setting up a second laptop** meant reinstalling Git, recreating the git-ignored `.env` and virtual environment, and learning that `pytest` has to run from the project root so relative paths like `frontend/` resolve.
- **I pushed straight to `main` by accident**, which is how the repo got branch protection. Every change goes through a PR now, which is how it should have been from the start.

---

### What I'd tell myself at the start

**The expensive bugs were the silent ones.** The cache, the missing tables, the rate limit that did nothing: none of them threw an error anyone could see. What caught them was poking the real thing from the outside, with smoke tests and plain `curl`.

**Local proves very little.** Every production failure here passed locally. Each hosted service removed one assumption I didn't know I'd made.

**Data problems hide in numbers you never print.** The PNG shortcut, the duplicate frames and the broken file were all found only after I added a check that printed something.

**If it doesn't show progress, you can't tell slow from stuck.** I'd have saved whole evenings with a progress bar.