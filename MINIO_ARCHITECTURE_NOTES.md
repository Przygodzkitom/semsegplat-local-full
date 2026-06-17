# Why MinIO Is Not Redundant in This Pipeline

## The Wrong Assumption

MinIO looks redundant at first glance because Label Studio can serve images from local files and stores annotations in its own SQLite database. It seems like adding an extra storage layer for no reason.

This assumption is wrong.

---

## What MinIO Actually Does

MinIO is not just storage — it is the **decoupling layer** between Label Studio (annotation tool) and the training pipeline (Python dataloaders).

```
User uploads image
        │
        ▼
   Streamlit UI ──────────────────► MinIO bucket
                                         │
                                    images/
                                    annotations/   ◄── Label Studio writes here
                                         │              on every annotation submit
                                         ▼
                                   Dataloader
                                   (training_brush.py /
                                    training_polygon.py)
```

### Label Studio side
When a user submits an annotation in Label Studio, the **export storage** (configured as S3/MinIO) automatically writes a JSON file to MinIO. This happens without any extra code — it is built into Label Studio's storage API.

### Training pipeline side
The dataloaders (`BrushDataset`, `LabelStudioMinIODatasetNumeric`) list annotation files in MinIO, download the JSON, and download the corresponding image. They have no knowledge of Label Studio at all.

### The key property
Neither side knows about the other. Label Studio pushes; the dataloader pulls. MinIO is the neutral intermediary.

---

## What Happens Without MinIO

Removing MinIO removes the automatic export mechanism. Annotations now only exist inside Label Studio's SQLite database. The training pipeline has no way to read them without one of these workarounds:

| Option | Description | Downside |
|---|---|---|
| LS API in dataloader | Dataloader calls Label Studio export API directly | Couples training code to Label Studio internals |
| Pre-training export step | Export annotations to local JSON files before training starts | Extra step, requires coordination |
| Local file export storage | Configure Label Studio to write annotation files to a local folder | `/api/storages/export/localfiles/` endpoint returned 404 in LS 1.20 |

All three options add complexity that MinIO was absorbing invisibly.

---

## The Architectural Lesson

> MinIO's value is not that it stores files — it is that it lets two components (Label Studio and the training pipeline) evolve independently without knowing about each other.

This is the same pattern as a message queue: the producer (Label Studio) and consumer (dataloader) are decoupled. MinIO is a simple, reliable implementation of that pattern using an S3-compatible API that both Label Studio and boto3 speak natively.

The "redundancy" was the point.

---

## Conclusion

Keep MinIO in the pipeline. The cost is one extra Docker container (~300 MB, idle most of the time). The benefit is a clean, automatic data flow where:

- Annotating in Label Studio is sufficient — no manual export step
- Training code is simple file reads, not API calls
- Either side can be replaced or upgraded without touching the other
