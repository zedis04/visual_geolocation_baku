# Take-Home Technical Challenge: Ground Level Visual Geolocation 

## 1. Objective

Build a model that, given a photograph taken in Baku, predicts the location where the photo was captured using only the visual content of the image. You are responsible for the full pipeline: data collection, model training, benchmark construction, evaluation, and reporting.

## 2. Problem Description

Many photographs are shared without location metadata. The objective of this task is to develop a system that infers the location of a photo from pixel data alone. The model is expected to perform across both well-known landmarks and ordinary urban scenes. The intended use case is general-purpose visual geolocation, similar in spirit to systems like Picarta.

## 3. Scope

- Geographic area: Baku city.
- Photo type: outdoor ground-level photographs (street photographs, handheld shots). The model shall accept only ground-level photographs as input; indoor, aerial, or satellite images can not be used at inference.
- Conditions: any time of day, any season, any weather.
- Applicability: although the system is built for Baku, the approach must be designed so that it can be easily adapted to other cities in Turkey or Azerbaijan by retraining on data from those cities, without redesigning the architecture. You may use aerial or satellite imagery during training if it helps you build the system, but the model must accept only ground-level photographs at inference.

If you choose to restrict the scope further (for example, to a subset of districts), state and justify this in your report.

## 4. Input and Output Specification

### 4.1 Input

A single RGB image file in JPG or PNG format. The image is a ground-level photograph. The model must operate on pixel data only. EXIF metadata, filenames, and any non-pixel signals are not available at inference time.


![Sample Input](https://i.pinimg.com/originals/94/e2/92/94e292050f9f7be037501970f29be65b.jpg "Sample Input Image Taken in Baku, Image by @fidanosmanova0219 via Pinterest")



### 4.2 Output

A JSON object containing the predicted location. You may choose one of the following representations:

- Geographic coordinates: latitude and longitude in WGS84.
- Named place classification: a label drawn from a taxonomy you define (district, neighborhood, or named landmark).

Pin-point accuracy is not expected. A common approach is to divide the target area into a grid (or use any other coarse-to-fine spatial discretization) and predict either the grid cell or the cell centroid. Other reasonable formulations are acceptable. Document whichever you choose.

The output must include a confidence score and may include a top-K list of alternative predictions. You may include both representations (coordinates and named place) if your system produces both, but only one is required.

Examples:

Coordinate-based output:

```json
{
  "lat": 40.3593,
  "lon": 49.8348,
  "confidence": 0.71,
  "top_k": [
    {"lat": 40.3593, "lon": 49.8348, "confidence": 0.71},
    {"lat": 40.3697, "lon": 49.8421, "confidence": 0.12}
  ]
}
```

Named-place output:

```json
{
  "place": "Sahil Plaza",
  "confidence": 0.58,
  "top_k": [
    {"place": "Sahil Plaza", "confidence": 0.58},
    {"place": "ISR Residence", "confidence": 0.16},
    {"place": "Narimanov District", "confidence": 0.07}
  ]
}
```

The output schema must be documented in the submission.

## 5. Dataset Requirements

You are responsible for assembling the data needed to build and evaluate your system. The architecture is not constrained to single-image input during training; you may use any data modality, pairing, or auxiliary signal you find useful. But during inference, only RGB image will be provided

1. **Sources.** You may collect data from any source. The submission must list the sources used.
2. **Geographic coverage.** The dataset must cover diverse regions of Baku.
3. **Diversity.** The dataset must reflect varying conditions (angle, lighting, time of day, season). Near-duplicates do not count toward diversity.

## 6. Benchmark Requirements

You are required to construct your own held-out benchmark.

1. **Independence.** Benchmark images must not appear in near-duplicate form, in any training or validation split. Document how you ensured this.
2. **Difficulty distribution.** The benchmark must include easy cases (clearly identifiable landmarks), medium cases (recognizable but non-iconic streetscapes), and hard cases (generic residential areas, ambiguous scenes, low-light photographs).
3. **Geographic distribution.** The benchmark must span multiple districts of Baku.
4. **Justification.** The submission must include a written justification of why performance on this benchmark is a fair indicator of real-world performance.

## 7. Evaluation Metrics

The submission must report, at minimum:

- If your output uses coordinates: mean and median great-circle distance error in meters, and the proportion of predictions within 100 m, 500 m, 1 km, and 5 km of the ground truth.
- If your output uses named-place classification: top-1 and top-5 accuracy, and per-class precision and recall.
- A per-district performance breakdown.
- A qualitative analysis of at least five failure cases, with discussion of likely causes.
- You may alter above guidelines on evaluation metrics, given you justify the approach.

Confidence calibration and inference latency are optional but recommended.

## 8. Real-World Performance

The system is expected to be useful on real photographs encountered in the wild — not only on images drawn from the same source distribution as the training data. Submissions that demonstrate robustness across viewpoints, lighting, and unfamiliar scenes are valued more highly than submissions that overfit to a clean, narrow data distribution. Keep this in mind when designing your data collection, training procedure, and benchmark.

## 9. Deliverables

The submission must include:

1. Source code for data preparation, training, evaluation, and inference.
2. An inference script with a documented command-line interface, for example:
   ```
   python predict.py --image path/to/photo.jpg
   ```
   producing the JSON output specified in Section 4.2.
3. Trained model weights, or instructions to obtain them.
4. A dataset summary describing image counts, distribution, sample images, and the sources used.
5. The benchmark dataset, packaged separately, with ground-truth labels (if used in system). A typical layout is a folder of images plus a single labels file (CSV or JSON) mapping each filename to its ground-truth location. Document the exact format you used.
6. A written report (PDF or markdown) covering the approach, complete description of system/model architecture, researched papers/sources, evaluation results, failure analysis, and known limitations.
7. A recorded video demo of system.
8. A README documenting environment setup and reproduction steps.

## 10. Constraints

1. Any open-source library, pre-trained model, public dataset, map service, or other resource may be used for training and data collection. The sources used must be disclosed in the submission.
2. Commercial APIs may be used for data collection (for example, geocoding) but not at inference time. The model must produce predictions from pixel input alone.
3. The system must be runnable on a single machine with a single GPU or CPU at inference time. Hardware requirements must be documented.
4. The system shall be able to train and operate on hardware freely available on mainstream cloud services. Lower performance/architecture choice due to hardware limitation is acceptable, assuming the hardware used is utilized efficiently. 
5. No metadata leakage at inference. The model must operate on pixel data only.
6. The use of AI assistants is permitted for this challenge, provided you maintain full command over every aspect of the work delivered, down to the minute details. 

## 11. Evaluation Criteria

Submissions will be evaluated on the following criteria, in approximate order of weight:

1. Quality of reasoning, clarity, robustness and originality of approach, treatment of failure cases, and self-assessment in the report.
2. Quality of the dataset and benchmark: coverage, diversity, and fairness.
3. Model performance on the submitted benchmark.
4. Engineering quality: reproducibility, code clarity, and correctness of the inference script.

## 12. Timeline and Contact

- **Deadline:** Within 3 weeks of receiving the task.
- **Questions:** You may direct any questions about the task to HR deparment.

## 13. Submission and Partial Submissions

Submit a link to a repository (for example, GitHub) or a format of your choosing, containing the deliverables listed in Section 9. Include an estimate of the time spent on the task.

**Partial submissions are encouraged.** If you are not able to complete the full pipeline, submit what you have. The reasoning, the design decisions, the data collection process, and the failure analysis are all evaluated independently of final model performance. A thoughtful partial submission with a clear write-up is preferred over no submission, and may score higher than a complete-but-shallow submission. We are interested in how you think about the problem, not only the final numbers.