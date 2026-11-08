"""Run the test_* functions of one test file, one at a time, without pytest.

Each test is timed and a failure does not stop the others.
    python tests/run_tests.py tests/test_physics.py
"""
import importlib.util
import os
import sys
import time
import traceback

path = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(path))
spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tests = [(name, fn) for name, fn in vars(mod).items() if name.startswith("test_") and callable(fn)]
failed = 0
for name, fn in tests:
    doc = (fn.__doc__ or "").strip().splitlines()
    start = time.time()
    try:
        fn()
        status = "PASS"
    except Exception:
        status = "FAIL"
        failed += 1
        err = traceback.format_exc()
    print(f"{status}  {name:45s} {time.time() - start:7.2f} s   {doc[0] if doc else ''}")
    if status == "FAIL":
        print("      " + err.replace("\n", "\n      "))
print(f"\n{len(tests) - failed}/{len(tests)} passed")
sys.exit(1 if failed else 0)
