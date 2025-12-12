import os, joblib

md = "models"
print("models folder:", os.path.abspath(md))

if not os.path.isdir(md):
    print("NO models/ folder found")
else:
    for f in sorted(os.listdir(md)):
        p = os.path.join(md, f)
        if os.path.isfile(p):
            print("\nFILE:", f)
            try:
                obj = joblib.load(p)
                print("  loaded type:", type(obj).__name__)
                named = getattr(obj, "named_steps", None)
                if isinstance(named, dict):
                    print("  named_steps:", list(named.keys()))
                print("  has .score() ?", hasattr(obj, "score"))
            except Exception as e:
                emsg = repr(e)
                print("  load failed:", emsg[:200])
