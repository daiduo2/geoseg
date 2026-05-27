---
name: module-demo
description: Run a module's demo.py and show the last 30 lines of output. Use to verify a module still works after edits.
disable-model-invocation: true
---

Run the demo for module: $ARGUMENTS

```bash
python -m geoseg.modules.$ARGUMENTS.demo
```

Show the last 30 lines of stdout/stderr and report PASS/FAIL based on exit code.
If the module name is invalid, list available modules:
```bash
ls geoseg/modules/
```
