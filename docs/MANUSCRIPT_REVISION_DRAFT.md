# ARGUS — Concise Manuscript Revision (I2ASPO 13-Page Limit)

> **Purpose and page constraint**
>
> This is the condensed replacement draft for the manuscript body. It is designed around the I2ASPO limit of **13 pages maximum**, excluding the cover, table of contents, bibliography/list of references, and other explicitly exempt front/back matter. The page count applies to the formatted manuscript, not to this Markdown file.
>
> **Target page allocation (13 pages total):** Abstract 0.5 page; Chapter 1 2.0 pages; Chapter 2 3.0 pages; Chapter 3 7.5 pages. Tables and equations are included in these allocations. Keep the final Word layout within this budget.
>
> **Scope:** Abstract and Chapters 1–3 only. Chapter 4/results and discussion are intentionally omitted. This draft does not claim that proposed tests have been completed.
>
> **Formatting note:** Actual pagination depends on the official I2ASPO template, font, margins, line spacing, and figure/table sizes. Apply the official template first, then make a final page-count check. Do not shrink fonts or margins below the competition rules to force compliance.

---

# ABSTRACT

Urban intersections experience changing traffic demand in which vehicle classes differ in their use of road space. Fixed-time signal plans do not directly respond to these observed variations, while emergency-vehicle prioritization requires reliable evidence before a signal phase is changed. This study presents ARGUS (Adaptive Real-time Green-time Update System), a software-based traffic-signal simulation for a four-approach intersection that combines multi-source vehicle detection, operational vehicle weighting, and dual-modal emergency-vehicle verification. The visual subsystem uses a trained YOLO26n model to detect and classify eight vehicle categories: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Detected vehicle counts are converted into an approach-level weighted traffic score using study-defined class weights. The controller distributes a total of 120 seconds of green time among the four approaches in proportion to these scores, subject to minimum and maximum green-time bounds. The emergency-verification subsystem analyses the soundtrack of a user-selected local video. Audio is processed in three-second mono segments at 16 kHz, represented by 1,024-dimensional embeddings from pretrained YAMNet, and classified by a Logistic Regression model into ambulance, firetruck, police, and traffic categories. Emergency priority is requested only when the visual emergency-class confidence and the aggregate audio emergency score both exceed their configured thresholds on the same approach. Yellow and all-red intervals are retained during signal transitions. The study describes the system architecture, implementation, and an experimental protocol for evaluating detection, timing allocation, and emergency verification. Since ARGUS is currently a software simulation and the completed experimental results are not presented in this manuscript draft, no claim of field effectiveness or measured traffic improvement is made.
Keywords: adaptive traffic signal; YOLO26n; passenger car unit; vehicle detection; YAMNet; emergency vehicle detection; dual-modal verification.


---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background of the Study

Signalized intersections divide available right-of-way among traffic streams approaching from different directions. Traffic demand varies over time, while vehicle categories differ in size and operating characteristics. Treating every detected vehicle as an identical unit can therefore provide an incomplete representation of traffic demand. The Passenger Car Unit (PCU) concept expresses vehicle classes relative to a passenger-car reference unit, although its values depend on traffic composition, road geometry, and operating conditions [7, 16, 22].

Computer vision enables roadside video to be processed for vehicle detection, classification, and counting. Previous studies have investigated computer-vision-based traffic monitoring and adaptive signal control [1–5, 14, 15]. However, detection quality depends on camera position, image resolution, lighting, occlusion, and the conditions represented in the training data. Model detections must consequently be treated as estimates rather than perfect observations.

Emergency vehicles create an additional control challenge. Visual detection may be affected by occlusion, while acoustic detection may be affected by background noise or sounds that resemble sirens. Research has explored audio-based and audio-visual emergency-vehicle detection [8–12]. Combining visual and audio evidence can provide a verification condition before an emergency-priority request is made.

This study develops ARGUS (Adaptive Real-time Green-time Update System), a software-based simulation of a four-approach intersection. Its visual subsystem uses a trained YOLO26n model to classify vehicles from configured video sources. Class counts are converted into weighted approach scores and used to allocate green time. Its audio subsystem analyses the soundtrack of a selected local video using pretrained YAMNet embeddings and a Logistic Regression classifier. Emergency priority is requested only when the visual and audio conditions exceed their configured thresholds on the same approach.

ARGUS is a research simulation, not a deployed road controller. It does not send commands to physical traffic-signal hardware. The study focuses on the integration and controlled evaluation of vehicle detection, weighted timing allocation, and dual-modal emergency verification.

## 1.2 Problem Statement

This study addresses four questions:

1. How can YOLO26n be integrated into a four-approach simulation to detect and classify vehicles from video sources?
2. How can class-specific operational weights be used to calculate approach-level traffic scores and allocate bounded green time?
3. How can visual emergency detections and audio classification be combined to verify an emergency event on the same approach?
4. How can the software simulation be evaluated in terms of vehicle detection, timing allocation, and emergency-verification behaviour under controlled scenarios?

## 1.3 Objectives of the Research

The objectives are to:

1. Develop a software simulation representing a four-approach intersection with configurable video sources.
2. Integrate YOLO26n to detect eight vehicle classes and obtain approach-level counts.
3. Calculate weighted traffic scores and allocate green time subject to configured limits.
4. Integrate YAMNet embeddings and Logistic Regression to classify audio from a selected local video's soundtrack.
5. Implement same-approach dual-modal emergency verification.
6. Define evaluation procedures for detection, timing, and emergency-verification performance.

## 1.4 Significance of the Research

**Theoretical significance.** The study provides an implementation framework combining multi-class vehicle detection, operational vehicle weighting, bounded signal timing, and dual-modal emergency verification. It does not propose a new YOLO architecture or a universal PCU standard.

**Practical significance.** The prototype can support controlled experiments and demonstrations of adaptive signal timing and emergency-event verification. It may also help identify issues involving video quality, class-detection errors, audio-source mapping, and threshold selection. Real-road deployment would require further safety engineering, local calibration, hardware integration, regulatory approval, and field testing.

## 1.5 Research Hypothesis

For a study whose primary statistical outcome is visual vehicle detection:

- **H0:** Under the defined test conditions, the YOLO26n-based detector does not meet the pre-established vehicle-detection performance criterion.
- **Ha:** Under the defined test conditions, the YOLO26n-based detector meets the pre-established vehicle-detection performance criterion.

The metric, acceptance criterion, test set, and statistical procedure must be specified before analysis. Separate hypotheses are required if audio accuracy or signal-response time is also tested inferentially.

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Theoretical Framework

### 2.1.1 Heterogeneous Traffic and Vehicle Weighting

Traffic flow, speed, and density describe different traffic characteristics. Flow is the number of vehicles passing a point per unit time; speed describes vehicle movement; density is the number of vehicles per unit road length. Image occupancy is the proportion of an image region covered by objects and should not be treated as equivalent to traffic density.

The PCU concept converts heterogeneous vehicle categories into passenger-car-equivalent units. PCU estimates vary with traffic composition and road conditions [7, 16, 22]. Therefore, values from a particular study should not be assumed to apply universally.

ARGUS calculates an approach-level weighted traffic score:

\[
W_i = \sum_{c=1}^{C} N_{i,c}w_c
\]

where \(N_{i,c}\) is the detected count of class \(c\) on approach \(i\), and \(w_c\) is its configured operational weight.

| Vehicle class | ARGUS operational weight |
|---|---:|
| Ambulance | 1.00 |
| Bus | 2.00 |
| Car | 1.00 |
| Fire engine | 2.00 |
| Military vehicle | 2.00 |
| Motorcycle | 0.33 |
| Police car | 1.00 |
| Truck | 2.00 |

These are the current simulator parameters, not a locally calibrated or universally applicable PCU standard. The cited motorcycle-PCU research [7] does not validate every value in this table. The weighted score is an operational input to the timing algorithm, not a direct measurement of queue length or density.

### 2.1.2 Computer Vision for Vehicle Detection

Object-detection systems identify objects in image frames and return class labels, bounding boxes, and confidence scores. Computer vision has been applied to traffic monitoring and adaptive signal systems [1–5, 14, 15]. ARGUS uses a trained YOLO26n model with eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck.

The current GUI uses a default confidence threshold of 0.15 and NMS IoU threshold of 0.60. Frames are processed at 640 × 480 with an inference image size of 640. A five-frame median filter smooths short-term fluctuations in class counts. This filtering does not establish persistent vehicle identity or cross-camera tracking. Detection performance may vary with lighting, occlusion, camera angle, and image quality; it must be measured on a labelled test set.

### 2.1.3 Acoustic Classification

Emergency-vehicle sirens can provide acoustic evidence when visual observations are incomplete. Prior studies have examined acoustic emergency-vehicle detection and audio-visual methods [8–12]. The active ARGUS GUI uses pretrained YAMNet as an audio feature extractor and Logistic Regression as the project-specific classifier.

The classifier identifies four categories: ambulance, firetruck, police, and traffic. The aggregate emergency score is:

\[
S_{audio}=P(ambulance)+P(firetruck)+P(police)
\]

The scores are model outputs and should not be called calibrated probabilities unless calibration has been performed. The audio source is the soundtrack of a user-selected local video, processed in three-second mono segments at 16 kHz. This is not an independent MP3 input or live microphone system. The current GUI inference path uses YAMNet embeddings and Logistic Regression; MFCC/librosa is not an active parallel detection branch.

### 2.1.4 Adaptive Signal Timing and Emergency Priority

Adaptive signal systems adjust timing in response to observed traffic conditions, unlike fixed-time plans [2–5, 14, 15]. ARGUS uses the weighted scores to distribute a total green-time budget among four approaches.

A basic proportional allocation is:

\[
G_i^*=G_{total}\frac{W_i}{\sum_{j=1}^{4}W_j}
\]

when the sum of approach scores is greater than zero. The current configuration uses a total green allocation of 120 seconds, with each approach constrained to 10–40 seconds. The actual implementation applies bounded redistribution; therefore, the equation above expresses the proportional principle, not the complete bounded algorithm. If all approach scores are zero, the system uses equal allocation.

Yellow and all-red intervals are 3 and 1 seconds respectively. They are additional to the green-time budget. A normal four-approach rotation therefore has a nominal duration of:

\[
T=120+4(3+1)=136\text{ seconds}
\]

Emergency priority is requested when the visual emergency confidence and aggregate audio emergency score both exceed 0.65 on the same approach. The comparison is strict (\(>\)). Signal transitions retain the configured yellow and all-red intervals. These thresholds are simulator settings, not field-calibrated safety values.

---

# CHAPTER 3 — RESEARCH METHOD

## 3.1 System Requirements and Characteristics

ARGUS is a software simulation of a four-approach intersection. Its functional requirements are to:

1. Receive video from configured CCTV, webcam, built-in, or local-video sources.
2. Detect and classify vehicles using YOLO26n.
3. Aggregate vehicle counts by class and approach.
4. Calculate weighted approach scores using the configured class weights.
5. Allocate 120 seconds of total green time with per-approach bounds of 10–40 seconds.
6. Maintain 3-second yellow and 1-second all-red transitions.
7. Analyse a selected local video's soundtrack using YAMNet and Logistic Regression.
8. Request emergency priority only when visual and audio scores both exceed 0.65 on the same approach.
9. Display video, counts, scores, signal states, and timers in the GUI.
10. Record system snapshots and export data for subsequent analysis.

The main implementation settings are summarized below.

| Component | Configuration |
|---|---|
| Intersection | North, East, South, West |
| Visual model | Trained YOLO26n |
| Vehicle classes | 8 classes listed in Section 2.1.2 |
| Confidence / NMS IoU | 0.15 / 0.60 default |
| Frame / inference size | 640 × 480 / 640 |
| Count smoothing | Five-frame median |
| Green allocation | 120 seconds total; 10–40 seconds per approach |
| Clearance intervals | Yellow 3 seconds; all-red 1 second |
| Audio pipeline | YAMNet embedding + Logistic Regression |
| Audio input | Selected local-video soundtrack |
| Audio window | 3 seconds, mono, 16 kHz |
| Audio categories | ambulance, firetruck, police, traffic |
| Emergency thresholds | Visual 0.65; audio 0.65 |

When multiple local videos are selected, the current GUI uses the first local video found in approach order as the soundtrack source. It does not provide a separate soundtrack analysis for every local-video lane. This constraint must be reflected in the experimental setup.

## 3.2 System Architecture and Algorithm Design

### 3.2.1 Input

Visual input consists of video frames from configured sources for the four approaches. Depending on the test setup, sources may be CCTV streams, webcams, built-in sources, or local videos. Audio input is the soundtrack extracted from a selected local video. It is not a separate MP3 upload or a live microphone feed.

### 3.2.2 Process

The system processes input as follows:

1. Video frames are passed to YOLO26n for vehicle detection and classification.
2. Class counts are aggregated by approach and smoothed using a five-frame median.
3. Each class count is multiplied by its configured weight to calculate the approach score \(W_i\).
4. The timing module distributes the 120-second green budget using bounded proportional allocation, with 10- and 40-second limits. If all scores are zero, it uses equal allocation.
5. The selected local video's soundtrack is extracted, converted to mono, resampled to 16 kHz, and divided into three-second segments.
6. YAMNet generates a 1,024-dimensional embedding for each segment. Logistic Regression classifies the embedding into four audio categories.
7. The emergency audio score is calculated by summing the ambulance, firetruck, and police scores.
8. The verification module checks whether visual emergency confidence and audio emergency score both exceed 0.65 on the same approach.
9. If verified, the system requests priority for that approach while retaining yellow and all-red transitions. Otherwise, normal weighted timing continues.

### 3.2.3 Output

The GUI displays video feeds, vehicle counts, weighted scores, signal states, countdown timers, and audio-classification information. Research snapshots can be exported as CSV for later analysis. These records represent system observations at logging times; they are not unique-vehicle trajectories or cross-camera identity records.

## 3.3 Software Implementation and Module Integration

### 3.3.1 Vehicle Detection

The visual module loads the trained YOLO26n checkpoint and detects the eight configured vehicle classes. The confidence and NMS IoU thresholds are adjustable through the GUI within their configured ranges. Five-frame median filtering is applied to reduce short-term fluctuations in class counts. The emergency visual classes are ambulance, fire engine, and police car.

The final paper must identify the checkpoint and report its training dataset, split, training settings, and independent test metrics only when those records are available. Do not infer training performance from inference code.

### 3.3.2 Audio Classification

The audio module extracts the soundtrack of the selected local video. Three-second mono segments are resampled to 16 kHz and passed to pretrained YAMNet. Each embedding has 1,024 dimensions and is classified by the Logistic Regression artifact `GUI/audio_models/best_audio_classifier_v2.joblib` into ambulance, firetruck, police, or traffic.

The project training discussion previously reported Logistic Regression with \(C=100\), validation Macro F1 of 92.17%, and test accuracy of 86.70%. Include these values only after checking the original training outputs, data split, and metric calculations. The inference module loads the serialized model; it does not independently prove the training configuration or metrics.

### 3.3.3 Weighted Traffic Estimation and Signal Control

Detected class counts are multiplied by the operational weights in Section 2.1.1 and summed for each approach. The bounded allocation module distributes 120 seconds of total green time, with a 10–40-second limit for each approach. Yellow and all-red intervals are 3 and 1 seconds and are additional to the green allocation. The nominal normal rotation is 136 seconds.

The weights are simulator parameters. They must not be described as universally valid PCU values without local calibration. The bounded redistribution behaviour should be checked against the current implementation, especially when one or more approaches reach a timing bound.

### 3.3.4 GUI and Data Logging

The GUI is implemented using CustomTkinter. It displays camera or video feeds, vehicle counts, weighted traffic information, signal states, timers, and audio scores. It provides simulation controls and source selection. The logging function records snapshots of system state and supports CSV export. The exported data should not be described as unique-vehicle tracking data.

### 3.3.5 Module Integration

The visual pipeline supplies approach-level vehicle counts to the weighting and timing modules. The audio pipeline supplies an aggregate emergency score from the selected local video's soundtrack. The emergency-verification module combines these outputs and requests priority only when both thresholds are exceeded on the same approach. Signal transitions retain the configured yellow and all-red intervals. ARGUS remains a software simulation and is not connected to physical traffic-signal controllers.

## 3.4 Experimental Setup and Testing Scenarios

The following items define the minimum evaluation protocol; they are not statements that the tests have already been conducted.

**Environment and input configuration.** Record computer specifications, operating system, Python and package versions, model/checkpoint identifiers, and the source used for each approach. For audio tests, identify the selected local video and its assigned approach.

**Vehicle detection.** Prepare a labelled test set separate from training and validation data. Report the number of images or frames, class distribution, annotation method, and evaluation conditions. Measure precision, recall, F1-score, and mAP using a stated IoU convention. Report per-class results.

**Weighted timing.** Define traffic scenarios with known vehicle counts and class compositions. Verify each weighted score and record the four green durations. Check that every duration remains within 10–40 seconds and that the four green allocations sum to 120 seconds. If the implementation uses a fallback for zero demand, verify that behaviour separately.

**Dual-modal verification.** Test positive and negative cases for ambulance, firetruck, and police audio classes. Include ordinary traffic audio and cases where one modality is absent or below threshold. Record both scores, approach mapping, verification decision, and decision time. A positive dual-modal case requires both scores to exceed 0.65 on the same approach.

**Signal transition.** Record the signal state and transition durations before, during, and after a priority request. Verify the yellow and all-red intervals. Describe recovery or time compensation only if the mechanism exists in the implementation and is directly tested.

Environmental robustness claims require actual tests under defined lighting, camera-angle, occlusion, image-quality, and audio-noise conditions. Do not claim robustness for conditions that were not tested.

## 3.5 Performance Metrics

Use metrics that directly answer the research questions:

1. **Detection:** precision, recall, F1-score, and mAP, with per-class reporting.
2. **Counting:** absolute count error or mean absolute error against manual annotations, if counting accuracy is evaluated.
3. **Timing allocation:** calculated weighted scores, green-time bounds, and sum of green durations.
4. **Emergency verification:** confusion matrix, precision, recall, F1-score, false-positive rate, and false-negative rate.
5. **Response time:** elapsed time between a clearly defined emergency-input event and the verification or phase-transition event.
6. **Transition compliance:** measured yellow and all-red durations against the configured 3- and 1-second intervals.

Delay, queue reduction, throughput, emergency travel time, and recovery equilibrium require a separate traffic-flow measurement method and a suitable baseline. They must not be inferred from vehicle counts or timing allocations alone.

---

# Author checks before submission

Keep this checklist outside the counted manuscript pages or use it as an internal editing note; remove it from the final competition paper if the template does not permit author notes.

- Confirm whether the actual research setup is software-only, a scaled physical prototype, or both. This draft describes the currently audited software implementation.
- Confirm the exact video sources used in experiments.
- Verify the audio dataset, class counts, train/validation/test split, and original model metrics before reporting 92.17% Macro F1 or 86.70% test accuracy.
- Verify the YOLO26n checkpoint, dataset, and independent test metrics.
- Decide whether operational vehicle weights remain assumptions or will be locally calibrated. Do not attribute the entire table to reference [7].
- Confirm the exact bounded redistribution and emergency-recovery behaviour in code.
- Keep claims about congestion, queue length, throughput, and emergency travel time out of the abstract and conclusions unless measured.
- Check every in-text citation against the bibliography and the exact claim it supports.

## Editorial compression log

Compared with the earlier long draft, this version removes repeated system descriptions, combines overlapping background and significance statements, shortens literature summaries to the concepts directly used by ARGUS, consolidates implementation settings into one table, and reduces the experimental section to essential protocols and metrics. It preserves the Abstract and Chapters 1–3 while excluding Chapter 4 and all unmeasured results.

**Page-count verification is still required in the official I2ASPO Word template.** If the body exceeds 13 pages after formatting, shorten prose and reduce nonessential tables/figures first; do not alter mandatory template typography or margins.
