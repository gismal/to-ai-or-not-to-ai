# Problems we faced

Every project has a version of its history where everything went to plan. This isn't that version.

These are the bugs, dead ends and "how did this ever work" moments from building AI or Not AI, written down while I still remember how much each one hurt. Each one comes with what broke, why, how I found it, and what I'd tell my past self.

---

## Production: where "works on my machine" goes to die

### The 502 that only existed on Render

`/explain` worked perfectly on my laptop and failed every single time in production. Classic.

The culprit was a background task. After answering a request, the API logged the prediction to the database, but it used the request's own database session, which was already closed by then. That killed the connection, and Render's proxy then happily reused the dead connection for the next request.

The fix: the logging task now opens its own session and swallows its own errors. Logging a prediction should never be able to break the prediction.

**Lesson:** background tasks outlive the request that started them. Give them their own stuff.

### The table that never existed

Render's logs told me `prediction_logs` didn't exist. Which was news to me, because it existed just fine locally.

It turned out both of my Alembic migrations were empty. Locally the tables had been created some other way, so nothing ever looked wrong. Production, starting from a clean database, simply never got them.

I replaced the empty migrations with one proper baseline that creates all five tables, and the Dockerfile now runs `alembic upgrade head` before the server starts. (It was also running a second time inside the app's startup. It doesn't anymore.)

**Lesson:** a model describes a table, a migration actually builds it. A working local database proves nothing about a fresh one.

### Three deploys, three different ways to fail

Fixing the migrations broke the next deploy: Neon's `alembic_version` table still pointed at one of the migrations I'd just deleted. Dropping that table and redeploying fixed it.

Then, in the middle of all this, the database URL on Render got changed to plain `postgresql://`, the format `psql` uses. SQLAlchemy read that as "use psycopg2", which isn't installed, and the deploy fell over again. The app needs `postgresql+asyncpg://`. The driver is part of the URL, and the URL your terminal likes isn't the one your app likes.

And then Upstash, my hosted Redis, refused to connect at all: "Only 0th database is supported." My code had hard-coded database 1 for the cache, because locally you get sixteen databases for free. Upstash gives you one. The code now uses the URL exactly as given.

**Lesson:** managed services quietly remove assumptions you didn't know you were making. Rewriting history (like migrations) means every database that remembers the old history has to be told.

### Votes that couldn't be cast twice

Users couldn't vote on the same image twice, and some clients couldn't vote at all. Two separate problems were hiding behind one symptom:

- A unique constraint allowed exactly one vote per filename and source. Fine for a first draft, wrong for real use.
- The list of allowed clients didn't include the web playground or the browser extension, which happen to be the two places people actually vote from.

Dropping the constraint needed a hand-written migration, my first one, with `DROP CONSTRAINT IF EXISTS` because not every database had the constraint in the first place.

**Lesson:** autogenerate can't read your mind; writing a migration by hand is completely normal. And `IF EXISTS` saves you when your environments have quietly drifted apart.

### A rate limit that limited nothing

The demo key was supposed to be rate-limited. The setting existed, the limiter existed, and neither was attached to a single route. The smoke test's rate-limit check against production is what caught it.

**Lesson:** configuration isn't behaviour. If something matters, test it from the outside.

### One comma, one 500

The admin "uncertain predictions" page crashed even with the right key. The cause was a single trailing comma after `.limit(limit),`, which quietly turned the query into a tuple. A `# type: ignore` on that line had been hiding the warning that would have pointed straight at it.

**Lesson:** `# type: ignore` is a promise that you know better than the type checker. I didn't.

### The test that failed only on slow machines

The rate-limit test fired six requests and expected the sixth to be blocked within one second. On my machine it was. On a busy CI runner it sometimes wasn't. The limit is now a setting, and the test uses a per-minute limit that no runner is slow enough to miss.

**Lesson:** a test that depends on how fast your computer is will eventually meet a slower computer.

---

## The cache that never cached (my favourite one)

This one deserves its own section, because it was invisible and a passing test was actively covering for it.

The API caches results, so the same image doesn't get analysed twice. Except it always did. Every cache read failed because of `json.load(raw)` where it should have been `json.loads(raw)`. Because I made a small typo. One letter. The error was caught and logged as a warning, so nothing ever looked broken. It was just slower than it should have been, forever.

When I fixed that letter, the code behind it ran for the first time ever, and it crashed immediately (a duplicate argument). It also had a bug of its own: on a cache hit it returned the *first* upload's filename, so uploading the same picture as `holiday.jpg` would show the name of whatever someone had uploaded before.

The worst part: there was a test called `test_cache_hit_on_second_request`, and it passed the whole time. It mocked the entire inference service, so it never touched the cache at all. It only proved that the route faithfully passes along whatever the mock says.

The new test runs the real service and the real cache against a tiny fake Redis that stores strings, the way real Redis does. It would have caught all three bugs.

**Lesson:** a test proves only what it doesn't mock. And an exception that gets caught and logged is still a failure, just a quiet one.

---

## Training data: where the model tries to cheat

### PNG = AI, apparently

Before training, I checked what file formats each class used. In the face data, 98% of AI images were PNG, while 71% of real photos were JPEG. A model doesn't care *why* the classes differ; if "is it a PNG?" gets the right answer most of the time, it will happily learn that instead of anything about faces.

The fix is JPEG re-compression on every training image, so file format carries no signal at all.

**Lesson:** before training, check what's trivially different between your classes. If you don't find the shortcut, the model will.

### Twins in the test set

One dataset is built from video frames, and neighbouring frames are nearly identical. A random split puts one frame in training and its twin in testing, and suddenly the test score is measuring memory, not skill. Perceptual hashing found and removed the twins: 194 from validation and 799 from test.

I also held two datasets out of training completely (Photoshop-edited faces and deepfakes). They're used only for testing, to see how the model handles kinds of fakes it has never seen. That number is less flattering than the main score, and more honest.

**Lesson:** a random split isn't an independent split when your data has near-duplicates.

---

## Kaggle: the long nights

Training runs on Kaggle's free GPUs, and this is where I lost the most hours.

**Huge images.** A handful of images were up to 161 megapixels, and the hashing step decoded each one at full size, eight at a time. The run slowed to a crawl and the cell sat there showing `*`. Images over 50 MP are now skipped, and hashing works on a reduced-size copy.

**Slow, or stuck?** Picking images opened files one at a time on Kaggle's slow input disk. Worse, if one dataset was mostly too-small images, the code read nearly all of it before moving on. One step took over 85 minutes and printed nothing, so it looked exactly like a frozen notebook. Now it checks files in parallel, gives up on a source after 300 unusable files in a row, and shows progress.

**The truncated image.** After two hours of preparing data, training crashed two minutes in, on one corrupted file. It had passed the size check because that only reads the file's header. The hashing step *had* failed on it, but rows without a hash were kept anyway. Now anything that can't be fully decoded is dropped before training.

**The sleeping laptop.** Interactive Kaggle sessions need your browser connected, so closing the laptop ended the run. Committed runs ("Save & Run All") run on Kaggle's servers and don't care what my laptop is doing.

**Lesson:** any long loop needs visible progress, because slow and stuck look identical from the outside. And validate data the same way it'll be used: a header check isn't a decode.

---

## Smaller ones worth a mention

- **The extension asked for access to every website at install.** Now that's optional: it asks for one site when you analyse an image there, and for all sites only if you turn on passive mode. A fresh install can reach nothing but the API. Users trust that more, and so do store reviewers.
- **A new favicon refused to show up.** Browsers cache favicons by URL, so `favicon.png?v=2` fixed it for everyone.
- **ONNX export broke after a PyTorch update.** Exporting at opset 18 instead of 12 fixed it.
- **I pushed straight to `main` by accident**, which is how the repo got branch protection. Every change goes through a PR now, which is how it should have been from the start.

---

## What I'd tell myself at the start

**The expensive bugs were the silent ones.** The cache, the missing tables, the rate limit that did nothing: none of them threw an error anyone could see. What caught them was poking the real thing from the outside, with smoke tests and plain `curl`.

**Local proves very little.** Every production failure here passed locally. Each hosted service removed one assumption I didn't know I'd made.

**Data problems hide in numbers you never print.** The PNG shortcut, the duplicate frames and the broken file were all found only after I added a check that printed something.

**If it doesn't show progress, you can't tell slow from stuck.** I'd have saved whole evenings with a progress bar.