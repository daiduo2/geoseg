---
name: schema-bump
description: Walk through pydantic-schema change with consumer-sync + integration-test gate. Use when modifying vlm_client/prompts.py or any pydantic BaseModel used across modules.
disable-model-invocation: true
---

Schema change protocol (DESIGN.md §4):

1. Edit `geoseg/modules/vlm_client/prompts.py` (or other schema source)
2. Bump `VERSION` string in `prompts.py`
3. Find all consumers:
   ```bash
   grep -rln "PageOverview\|PanelReview\|SegmentationReview" geoseg/ tests/
   ```
4. Update each consumer to match new schema
5. Run integration test:
   ```bash
   pytest tests/test_integration_ph01.py -v
   ```
6. If test doesn't exist yet, create a skeleton that validates the schema round-trip

Do NOT commit until step 5 passes.
