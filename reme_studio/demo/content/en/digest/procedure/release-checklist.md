---
name: Reusable release checklist
description: Turn acceptance experience into documented operating steps.
memory_tags: ["aurora", "release", "procedure"]
---

# Release checklist

For Aurora's internal pilot. [[digest/wiki/people.md|Zhou]] performs the checks and Lin confirms scope. Source: [[daily/2026-09-18/review.md|the review]].

- [ ] Preserve a workspace file copy and check direct Markdown access.
- [ ] Check source links in [[digest/wiki/local-first.md|the architecture decision]].
- [ ] Test cross-directory retrieval with `local_first` and `release` tags.
- [ ] Stop the service and open existing files offline.
- [ ] Check derived-data recovery using [[digest/procedure/recovery.md|the recovery steps]].
- [ ] Record passed, failed, and untested checks in the acceptance record.
- [ ] Update [[digest/wiki/release.md|release status]] with original evidence.

The checklist captures reusable experience. Checking an item does not start a background task. See [[resources/pilot-results.md|the acceptance record]] for current results.
