# ARGUS manuscript revision draft (Abstract–Chapter 3)

> **Status:** working replacement text prepared from the uploaded `FORMAT FULL PAPER (2).docx`, the current Lampiu repository documentation, and the implementation facts already audited. This is a manuscript draft, not experimental results. Review every author-confirmation item before submission.
>
> **Scope:** Abstract and Chapters 1–3 only. Chapter 4 and all results/discussion sections are intentionally excluded. Do not interpret proposed test protocols below as tests already completed.

## Editorial decisions applied

1. Describe ARGUS as a **software-based four-approach traffic simulation**, not a deployed road controller.
2. Describe audio input as the soundtrack of a **user-selected local video**, not a separate MP3 input or live siren microphone.
3. Describe the active audio pipeline as pretrained YAMNet embeddings followed by a four-class Logistic Regression classifier. MFCC/librosa is not described as an active detection branch because that is not the current GUI inference path.
4. Use the eight implemented visual classes: Ambulance, Bus, Car, Fire engine, Military, Motorcycle, Police car, and Truck.
5. Describe vehicle weights as **study-defined operational weights**. Do not call the complete table a universal PCU standard.
6. Distinguish the 120-second total green allocation from clearance time. With four approaches and one 3-second yellow plus one 1-second all-red transition per approach, a normal full rotation is 136 seconds.
7. Remove unverified claims that the system reduces congestion, improves throughput, outperforms single-modal systems, restores traffic equilibrium, or reduces emergency travel time. Such statements require measured results.
8. Retain Chapter 4 as a later section to be written only after experiments and analysis are complete.

---

# ABSTRACT

Urban intersections experience changing traffic demand in which vehicle classes differ in their use of road space. Fixed-time signal plans do not directly respond to these observed variations, while emergency-vehicle prioritization requires reliable evidence before a signal phase is changed. This study presents ARGUS (Adaptive Real-time Green-time Update System), a software-based traffic-signal simulation for a four-approach intersection that combines multi-source vehicle detection, operational vehicle weighting, and dual-modal emergency-vehicle verification. The visual subsystem uses a trained YOLO26n model to detect and classify eight vehicle categories: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Detected vehicle counts are converted into an approach-level weighted traffic score using study-defined class weights. The controller distributes a total of 120 seconds of green time among the four approaches in proportion to these scores, subject to minimum and maximum green-time bounds. The emergency-verification subsystem analyses the soundtrack of a user-selected local video. Audio is processed in three-second mono segments at 16 kHz, represented by 1,024-dimensional embeddings from pretrained YAMNet, and classified by a Logistic Regression model into ambulance, firetruck, police, and traffic categories. Emergency priority is requested only when the visual emergency-class confidence and the aggregate audio emergency score both exceed their configured thresholds on the same approach. Yellow and all-red intervals are retained during signal transitions. The study describes the system architecture, implementation, and an experimental protocol for evaluating detection, timing allocation, and emergency verification. Since ARGUS is currently a software simulation and the completed experimental results are not presented in this manuscript draft, no claim of field effectiveness or measured traffic improvement is made.

**Keywords:** adaptive traffic signal; YOLO26n; passenger car unit; vehicle detection; YAMNet; emergency vehicle detection; dual-modal verification.

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background of the Study

Signalized intersections must allocate limited right-of-way among traffic streams approaching from different directions. Traffic demand changes over time, and the vehicles within a traffic stream are heterogeneous in size and operating characteristics. A simple vehicle count treats a motorcycle, passenger car, bus, and truck as equivalent units even though they occupy different amounts of road space and may affect discharge conditions differently. Consequently, vehicle classification and an appropriate weighting method can provide a more informative representation of traffic demand than an unweighted count alone [7, 16, 22].

Computer vision offers a way to obtain traffic observations from roadside cameras without requiring a separate detector for every vehicle. Object-detection models can identify vehicle categories in video frames, after which detections can be aggregated into approach-level counts. Previous studies have investigated computer-vision-based traffic monitoring and adaptive signal control [1–5, 14, 15]. The performance of a vision-based system nevertheless depends on camera placement, image quality, lighting, occlusion, model accuracy, and the operating conditions represented in its training data. Therefore, a camera-based estimate should be treated as an observation produced by a model rather than a perfect measurement of all vehicles present.

One approach to representing heterogeneous traffic is the Passenger Car Unit (PCU) concept, which expresses different vehicle categories relative to a passenger-car reference unit. PCU values depend on traffic composition, road geometry, operating conditions, and the method used to estimate them [7, 16, 22]. For this reason, a value adopted in one study should not automatically be treated as universal. In ARGUS, class weights are used as operational parameters for the simulation. They convert detected vehicle counts into a weighted approach score used by the timing algorithm. These parameters are not presented as a locally calibrated PCU standard for every road in Yogyakarta.

Another challenge at signalized intersections is the passage of emergency vehicles. Visual detection may be limited when an emergency vehicle is partially or fully occluded by other traffic. Acoustic information can provide a complementary signal, and previous research has examined audio-based emergency-vehicle detection and audio-visual approaches [8–12]. However, an audio classifier may also respond to sounds that resemble sirens, while a visual detector may misclassify an object. Combining evidence from two modalities can therefore be used as a verification condition rather than relying on a single detection signal.

This study develops ARGUS (Adaptive Real-time Green-time Update System), a software-based simulation for a four-approach intersection. Its visual subsystem uses a trained YOLO26n model to classify vehicles from camera or video sources. The resulting vehicle counts are weighted using study-defined class parameters and used to allocate green time across the four approaches. Its audio subsystem analyses the soundtrack of a selected local video using pretrained YAMNet embeddings and a Logistic Regression classifier. The system requests emergency priority only when the visual and audio conditions are both satisfied for the same approach.

ARGUS is intended for controlled software simulation and research. It does not send commands to physical traffic-signal controllers and is not an operational traffic-management installation. Its purpose is to provide a testable software framework in which vehicle detection, weighted traffic estimation, signal timing, and dual-modal emergency verification can be examined under defined scenarios.

## 1.2 Problem Statement

This study addresses the following research questions:

1. How can a multi-class YOLO26n vehicle-detection model be integrated into a four-approach traffic simulation to obtain vehicle counts for each approach?
2. How can study-defined vehicle-class weights be applied to detected counts to calculate an approach-level weighted traffic score and allocate bounded green time?
3. How can visual emergency-vehicle detections and audio classification scores be combined to verify an emergency event on the same approach before requesting signal priority?
4. How can the implemented software simulation be evaluated in terms of vehicle-detection performance, timing allocation, and dual-modal emergency-verification behaviour under controlled test scenarios?

The fourth question concerns evaluation to be conducted and reported using measured data. It must not be answered with assumed or unmeasured performance claims.

## 1.3 Objectives of the Research

The objectives of this study are:

1. To develop a software-based four-approach traffic simulation that receives video from configured camera, CCTV, webcam, or local-video sources.
2. To implement vehicle detection and classification using the trained YOLO26n model and obtain approach-level vehicle counts.
3. To calculate a weighted traffic score from vehicle counts and study-defined class weights, then distribute the green-time allocation proportionally while enforcing configured minimum and maximum durations.
4. To integrate an audio-classification pipeline that analyses the soundtrack of a selected local video using YAMNet embeddings and a Logistic Regression classifier.
5. To implement a dual-modal emergency-verification condition that requires both visual and audio scores to exceed their configured thresholds on the same approach.
6. To define controlled evaluation scenarios and metrics for assessing detection performance, timing behaviour, and emergency-verification behaviour.

The objectives describe system development and evaluation. Any conclusion that ARGUS reduces waiting time, improves intersection throughput, or shortens emergency-vehicle travel time must be supported by experimental results and an appropriate comparison baseline.

## 1.4 Significance of the Research

### 1.4.1 Theoretical Significance

This study provides a software framework for examining the integration of three concepts in a four-approach intersection simulation:

1. **Heterogeneous vehicle representation.** The system uses class-specific operational weights rather than treating every detected vehicle as an identical unit. This allows the effect of different weighting assumptions on approach-level demand scores to be examined.
2. **Vision-based traffic observation.** The study applies a trained object-detection model to classify multiple vehicle categories from video input and aggregate detections by approach.
3. **Dual-modal emergency verification.** The system combines visual emergency-vehicle detections with audio-classification evidence as a same-approach verification condition before requesting priority.

The contribution is an implementation and evaluation framework. It should not be described as a new universal PCU standard, a new YOLO architecture, or a validated traffic-control theory.

### 1.4.2 Practical and Applied Significance

The prototype may support controlled experiments and educational demonstrations concerning adaptive signal timing, vehicle classification, and emergency-vehicle verification. Its modular design can also help identify practical issues such as camera-source quality, class-detection errors, audio-source mapping, threshold selection, and signal-transition constraints.

The current implementation is not connected to public-road signal controllers. Accordingly, this study does not claim that the prototype can be directly installed by municipalities, create a coordinated green wave across a road network, or safely control a real intersection. Such applications would require additional engineering, safety analysis, field calibration, regulatory approval, and testing beyond the scope of this software simulation.

## 1.5 Research Hypotheses

The hypotheses should be linked to measurable outcomes. The following formulation is recommended if the primary statistical evaluation concerns vehicle-detection performance:

- **H0 (Null Hypothesis):** Under the defined test conditions, the YOLO26n-based vehicle-detection system does not achieve the pre-established performance criterion for vehicle classification and detection.
- **Ha (Alternative Hypothesis):** Under the defined test conditions, the YOLO26n-based vehicle-detection system achieves the pre-established performance criterion for vehicle classification and detection.

Before statistical testing, the research team must define the performance metric, acceptance criterion, test dataset, and statistical procedure. If the study intends to test response time, timing fairness, or emergency-verification accuracy as separate outcomes, those outcomes require their own operational definitions and hypotheses. Do not use the phrase “statistically significant” unless an appropriate statistical test and decision criterion are actually applied.

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Theoretical Framework

### 2.1.1 Traffic Flow Dynamics and Heterogeneous Characteristics

Traffic flow at an intersection can be described using related variables such as flow, speed, and density. These variables describe different properties and should not be used interchangeably. Flow represents the number of vehicles passing a reference point per unit of time. Speed represents the rate of movement of vehicles. Density represents the number of vehicles occupying a unit length of roadway at a given time. Camera-based image occupancy is a visual measurement related to the proportion of an image region covered by objects; it is not automatically equivalent to traffic density.

In heterogeneous traffic, vehicle categories differ in physical dimensions and operating behaviour. A raw count therefore does not represent the same approximate traffic demand for every possible vehicle mixture. The PCU concept provides a way to express vehicle classes relative to a passenger-car reference unit. Research has shown that PCU estimates depend on local traffic and geometric conditions, including motorcycle composition and intersection layout [7, 16, 22]. Official Indonesian road-capacity guidance should be consulted when selecting values for a particular road context [28].

In this study, the weighted traffic score for approach (i) is defined as:

[
W_i = sum_{c=1}^{C} N_{i,c},w_c
]

where:

- (W_i) is the weighted traffic score for approach (i);
- (N_{i,c}) is the number of detected vehicles belonging to class (c) on approach (i);
- (w_c) is the study-defined operational weight for class (c);
- (C) is the number of vehicle classes included in the calculation.

The current GUI uses the following operational weights:

| Vehicle class | Operational weight |
|---|---:|
| Ambulance | 1.00 |
| Bus | 2.00 |
| Car | 1.00 |
| Fire engine | 2.00 |
| Military vehicle | 2.00 |
| Motorcycle | 0.33 |
| Police car | 1.00 |
| Truck | 2.00 |

These values describe the current simulator configuration. They must not be attributed wholesale to a single reference or described as universally valid PCU equivalents. In particular, the cited motorcycle-PCU study [7] reports values under its own study conditions; it does not establish every weight in this table for the ARGUS test environment. If the research team later selects values from PKJI or another local standard, the table, calculation, and justification must be revised consistently.

The weighted score (W_i) is an operational demand indicator used by the ARGUS timing algorithm. It is not a direct measurement of queue length, delay, traffic density, or spatial occupancy.

### 2.1.2 Computer Vision and Vehicle Detection

Computer vision enables software to extract information from images and video. In intelligent transportation research, roadside-camera footage can be processed to detect, classify, and count vehicles [1–5]. Object-detection systems typically return bounding boxes, class labels, and confidence scores for objects identified in a frame.

ARGUS uses a trained YOLO26n model for eight vehicle classes: Ambulance, Bus, Car, Fire engine, Military, Motorcycle, Police car, and Truck. The configured default confidence threshold is 0.15, while the GUI provides a slider range from 0.05 to 0.80. The default non-maximum-suppression IoU threshold is 0.60, with a GUI range from 0.30 to 0.90. Inference uses an image size of 640, and the processing frame is resized to 640 × 480. A five-frame median filter is applied to smooth fluctuations in class counts.

The model's detections are observations made from each processed frame. A five-frame median filter can reduce short-term count fluctuations, but it does not establish persistent vehicle identity or track the same vehicle across different cameras. ARGUS should therefore be described as multi-source or multi-camera monitoring, not cross-camera identity tracking, unless a separate tracking implementation is added and evaluated.

Detection performance can be affected by occlusion, lighting, camera angle, image resolution, motion blur, and class similarity. Consequently, model performance should be evaluated on a labelled test set representative of the intended video conditions. General claims that an object detector is robust under all weather or lighting conditions should not be made without corresponding tests [19].

### 2.1.3 Acoustic Signals and Emergency-Vehicle Classification

Emergency vehicles may use distinctive siren sounds, which can provide an additional source of evidence when visual observations are incomplete. Audio-based emergency-vehicle detection has been studied using machine-learning and deep-learning methods [8–10]. Audio-visual approaches have also been investigated to combine visual and acoustic evidence [10–12].

The current ARGUS GUI uses pretrained YAMNet as an audio feature extractor. YAMNet produces a 1,024-dimensional embedding for an audio input window. A Logistic Regression classifier then maps the embedding to one of four project-specific categories: `ambulance`, `firetruck`, `police`, and `traffic`. The emergency score is calculated by summing the classifier scores for the three emergency categories:

[
S_{mathrm{audio}} =
P(mathrm{ambulance}) +
P(mathrm{firetruck}) +
P(mathrm{police})
]

The waveform supplied to this pipeline comes from the soundtrack of a user-selected local video. The GUI analyses the soundtrack in three-second windows after conversion to mono and resampling to 16 kHz. This implementation is distinct from a standalone MP3-input design and from the older prototype that directly evaluates the AudioSet `Siren` class.

The classifier's `predict_proba` outputs and their aggregate emergency score should be described as model scores. Unless a calibration procedure has been performed, they should not be described as calibrated real-world probabilities. The selected training configuration and reported validation metric must be attributed to the project's training record, and should be accompanied by the dataset version, split strategy, and test results when those records are available.

### 2.1.4 Adaptive Signal-Control Algorithm

A fixed-time signal plan uses predetermined phase durations. An adaptive approach changes timing according to observed traffic conditions. Previous research has investigated computer-vision-based signal control, proportional timing, and other adaptive strategies [2–5, 14, 15]. ARGUS implements a bounded proportional allocation based on the weighted scores calculated for its four approaches.

Let (G_{mathrm{total}}) denote the total green-time budget, and let (W_i) denote the weighted traffic score for approach (i). A basic proportional allocation is:

[
G_i^{*} =
G_{mathrm{total}}
rac{W_i}{sum_{j=1}^{4} W_j}
]

This expression is applicable when the sum of all approach scores is greater than zero. If all scores are zero, the implementation uses an equal allocation. The current configuration uses a total green allocation of 120 seconds across four approaches and bounds each approach's green duration between 10 and 40 seconds. Because the implementation enforces lower and upper bounds, the final allocation is a bounded proportional allocation rather than an unrestricted application of the equation above. The manuscript should not imply that simply clipping each independent value always preserves the 120-second total; the bounded redistribution procedure must be described according to the actual implementation.

Signal transitions include a 3-second yellow interval and a 1-second all-red interval. These clearance intervals are additional to the 120-second green allocation. For a normal four-approach rotation with four transitions, the nominal elapsed time is:

[
T_{mathrm{rotation}} =
120 + 4(3+1) = 136 mathrm{s}
]

This value describes the configured normal rotation and excludes additional time introduced by emergency preemption or other pauses.

Emergency-vehicle preemption is a separate priority mechanism. In ARGUS, the visual emergency condition and aggregate audio emergency score must both exceed 0.65 on the same approach. The comparison uses a strict greater-than condition ((>), not (geq)). A requested priority change must retain the configured yellow and all-red transition intervals. These thresholds are initial simulator parameters, not values calibrated for public-road operation.

---

# CHAPTER 3 — RESEARCH METHOD

## 3.1 System Requirements Analysis and System Characteristics

ARGUS is a software-based traffic-signal simulation designed to represent a four-approach intersection. It integrates video-based vehicle detection, weighted traffic estimation, bounded green-time allocation, audio classification, and dual-modal emergency verification. The system is intended for controlled research and does not directly control physical traffic-signal hardware.

### 3.1.1 System Requirements Analysis

The system has the following functional requirements:

1. Receive video input for the North, East, South, and West approaches from configured CCTV streams, webcams, built-in sources, or local video files.
2. Detect and classify vehicles using the trained YOLO26n model.
3. Aggregate detected vehicles by the eight supported classes and by approach.
4. Apply the configured operational class weights to calculate an approach-level weighted traffic score.
5. Allocate green time among four approaches using bounded proportional allocation, with a total green-time budget of 120 seconds and per-approach limits of 10–40 seconds.
6. Maintain the configured yellow and all-red transition intervals.
7. Analyse the soundtrack of a selected local video using the YAMNet-embedding and Logistic Regression pipeline.
8. Calculate an aggregate emergency audio score from the ambulance, firetruck, and police classifier outputs.
9. Request emergency priority only when visual and audio conditions exceed their thresholds on the same approach.
10. Display video, vehicle counts, signal states, timers, and audio scores through the GUI.
11. Record research snapshots and allow recorded data to be exported for subsequent analysis.

The system also has non-functional requirements: the interface should remain responsive during video and audio processing; the software dependencies should be documented; input-source failures should be visible to the operator; and recorded data should retain enough configuration context to support later analysis.

### 3.1.2 System Characteristics

The principal characteristics of the current implementation are:

| Component | Current implementation |
|---|---|
| Intersection representation | Four approaches: North, East, South, West |
| Visual model | Trained YOLO26n |
| Vehicle classes | Ambulance, Bus, Car, Fire engine, Military, Motorcycle, Police car, Truck |
| Default confidence threshold | 0.15 |
| Default NMS IoU threshold | 0.60 |
| Inference image size | 640 |
| Processed frame size | 640 × 480 |
| Count smoothing | Five-frame median |
| Traffic weighting | Study-defined operational class weights |
| Green-time budget | 120 seconds total across four approaches |
| Green-time bounds | 10–40 seconds per approach |
| Yellow interval | 3 seconds |
| All-red interval | 1 second |
| Audio feature extractor | Pretrained YAMNet |
| Audio classifier | Logistic Regression |
| Audio classes | ambulance, firetruck, police, traffic |
| Audio window | 3 seconds, mono, 16 kHz |
| Audio emergency threshold | 0.65 |
| Visual emergency threshold | 0.65 |
| Emergency verification | Both conditions above threshold on the same approach |

The system's input sources are configurable. The audio scenario currently supported by the main GUI uses the soundtrack of a local video assigned to an approach. When multiple local videos are assigned, the current GUI selects the first local video found in approach order as the soundtrack source; it does not provide an independent audio stream for every local-video lane. This constraint must be reflected in experimental setup and interpretation.

## 3.2 System Architecture and Algorithm Design

### 3.2.1 Input

ARGUS receives two categories of input.

**Visual input.** Video frames are obtained from the configured sources for the four approaches. Depending on the selected setup, a source may be a CCTV stream, webcam, built-in source, or local video. The video frames are passed to YOLO26n for vehicle detection and classification.

**Audio input.** The active audio scenario uses the soundtrack of a user-selected local video. The soundtrack is extracted by the video-audio module and supplied to the audio-analysis pipeline. It is not an independent MP3 upload field and should not be described as live microphone-based siren detection.

Input quality affects the resulting observations. Visual detection may be affected by camera angle, image resolution, lighting, occlusion, and stream stability. Audio classification may be affected by soundtrack quality, background noise, siren type, recording conditions, and the correspondence between the selected video and the approach being simulated.

### 3.2.2 Process

The processing sequence is as follows:

1. The GUI obtains frames from the configured approach video sources.
2. YOLO26n detects and classifies vehicles in each processed frame.
3. The system aggregates class detections for each approach and applies five-frame median smoothing to reduce short-term count fluctuations.
4. Each class count is multiplied by its configured operational weight. The weighted values are summed to produce (W_i) for each approach.
5. The bounded allocation module distributes the 120-second green-time budget according to approach scores, subject to the 10–40-second limits. If all approach scores are zero, the configured fallback is an equal allocation.
6. The video-audio module extracts the soundtrack from the selected local video. The audio is segmented into three-second windows, converted to mono, and resampled to 16 kHz.
7. YAMNet generates a 1,024-dimensional embedding for each audio window. The Logistic Regression classifier produces scores for ambulance, firetruck, police, and traffic.
8. The audio emergency score is calculated by summing the scores for ambulance, firetruck, and police.
9. The emergency-verification logic checks whether the visual emergency confidence and audio emergency score both exceed 0.65 for the same approach.
10. If the dual-modal condition is satisfied, the system requests priority for the corresponding approach while retaining the yellow and all-red transition intervals. Otherwise, the normal traffic-weighted allocation continues.
11. The GUI displays the current observations and signal state, while research snapshots can be recorded and exported.

The emergency thresholds are initial configuration values. Their suitability must be evaluated through controlled testing and must not be represented as field-calibrated.

### 3.2.3 Output

The system displays the configured video feeds, vehicle-class counts, weighted traffic scores, signal states, countdown timers, and audio-classification information. It also provides controls for simulation operation and input-source selection. Research data can be recorded as snapshots and exported in CSV format for later analysis.

The exported records should be interpreted as observations at the time of logging. They do not constitute unique vehicle trajectories or cross-camera identity records. The current implementation does not establish a persistent identity for a vehicle as it moves between approaches or cameras.

## 3.3 Software Implementation and Module Integration

### 3.3.1 Software Implementation

#### 3.3.1.1 YOLO26n-Based Vehicle Detection

The visual subsystem uses a trained YOLO26n checkpoint to detect eight vehicle classes: Ambulance, Bus, Car, Fire engine, Military, Motorcycle, Police car, and Truck. The default confidence threshold is 0.15 and the default NMS IoU threshold is 0.60. The GUI provides controls to adjust these values within their configured ranges. Frames are processed at 640 × 480, with a model inference image size of 640. A five-frame median filter is used to smooth the vehicle counts.

The detected class counts are passed to the weighted traffic-estimation module. The visual emergency set comprises Ambulance, Fire engine, and Police car. The system uses the detection confidence associated with these classes as the visual condition for emergency verification.

The manuscript should identify the checkpoint file and its provenance in the final submission. The training dataset, data split, augmentation settings, training environment, and independent test metrics should be stated only when supported by the actual training records.

#### 3.3.1.2 YAMNet-Embedding Audio Classification and Emergency-Sound Scoring

The main GUI audio pipeline uses pretrained YAMNet as a feature extractor and a trained Logistic Regression estimator as the project-specific classifier. The soundtrack is extracted from a user-selected local video. Audio is processed in three-second windows, converted to mono, and resampled to 16 kHz before feature extraction. YAMNet generates a 1,024-dimensional embedding, which is supplied to the Logistic Regression classifier.

The classifier returns scores for four categories: ambulance, firetruck, police, and traffic. The aggregate emergency score is calculated as:

[
S_{mathrm{audio}} =
S_{mathrm{ambulance}} +
S_{mathrm{firetruck}} +
S_{mathrm{police}}
]

The current inference artifact is `GUI/audio_models/best_audio_classifier_v2.joblib`. The project training record identifies Logistic Regression with (C=100) as the selected configuration from the candidate values examined. A validation Macro F1 score of 92.17% was previously reported in the project training discussion. These figures should be retained in the manuscript only after the original training outputs, dataset split, and metric calculation have been checked and documented. The inference module loads the serialized estimator; it does not retrain the model or independently establish its training configuration.

The current GUI does not use librosa-based MFCC extraction as a parallel active inference branch. Therefore, the subsection title and method description should not claim that MFCC is part of the deployed GUI classifier unless the code is changed and that additional branch is implemented and tested.

#### 3.3.1.3 Operational-Weight Traffic Estimation and Signal Timing

For each approach, detected vehicle counts are multiplied by the operational weights listed in Section 2.1.1. The resulting values are summed to calculate the weighted traffic score. The scores are used by the bounded allocation module to distribute a total green-time budget of 120 seconds across four approaches. Each approach is constrained to a green duration between 10 and 40 seconds. If all scores are zero, the implementation uses an equal allocation.

The yellow interval is configured to 3 seconds and the all-red interval to 1 second. These intervals are additional to the 120-second green-time budget. Under the normal four-approach rotation, the nominal elapsed duration is 136 seconds, excluding emergency overrides and other pauses.

The operational weights are parameters of the current simulation. They should not be presented as universally applicable PCU values or as locally validated values without a separate calibration study.

#### 3.3.1.4 Graphical User Interface and Research Data Logging

The GUI is implemented using CustomTkinter. It displays video sources, vehicle counts, weighted traffic information, signal states, countdown timers, and audio-classification scores. The launcher supports selecting sources for the four approaches. The main GUI also provides simulation controls and configurable detection parameters.

The logging functionality records periodic snapshots of the system state and supports CSV export. The exported data can be used to calculate evaluation metrics after the test protocol has been completed. A snapshot log should not be interpreted as a unique-vehicle dataset or a record of cross-camera trajectories.

### 3.3.2 Module Integration

ARGUS integrates the visual and audio pipelines with the traffic-estimation and signal-control modules. The visual pipeline produces class-level vehicle counts for each approach. These counts are transformed into weighted traffic scores and passed to the bounded green-time allocation module. The audio pipeline processes the soundtrack of the selected local video and produces an aggregate emergency score.

The dual-modal verification module combines the visual and audio conditions. Emergency priority is requested only when the visual emergency condition and aggregate audio emergency score both exceed 0.65 on the same approach. The signal transition retains the configured yellow and all-red intervals before the requested approach receives priority. If the dual-modal condition is not satisfied, the simulation follows its normal traffic-weighted allocation.

The GUI presents the resulting state and records research snapshots. Separate processing components are used for video and audio analysis to support interface responsiveness. ARGUS remains a software simulation; it is not connected to physical traffic-signal controllers.

## 3.4 Experimental Setup and Testing Scenarios — protocol to be completed

> **Important:** The following is a proposed structure for documenting the experiment. Replace bracketed fields with actual values and retain only scenarios that were genuinely conducted. Do not write “we tested” or report outcomes before the corresponding test has been performed.

### 3.4.1 Simulation Environment and Configuration

Report the actual computer specifications, operating system, Python version, package versions, YOLO checkpoint identifier, audio-classifier artifact identifier, video-source details, and GUI configuration. State whether each approach used CCTV, webcam, or local video. For audio tests, identify the local video used as the soundtrack source and the approach to which it was assigned.

### 3.4.2 Vehicle-Detection Evaluation

Prepare a labelled evaluation set with ground-truth vehicle classes and bounding boxes. State the number of images or frames, class distribution, annotation procedure, and separation from training/validation data. Evaluate precision, recall, F1-score, and mAP using a clearly specified IoU convention. Report per-class results as well as aggregate results. If only vehicle counts are evaluated, define the counting-error metric separately from object-detection metrics.

### 3.4.3 Weighted Traffic and Green-Time Allocation

Define low-, medium-, and high-demand scenarios using recorded or constructed vehicle-count inputs. State the exact class composition for each approach. Verify that calculated weighted scores match the configured operational weights. Record each approach's allocated green duration and check whether the minimum and maximum bounds are respected and whether the four green durations sum to 120 seconds.

Do not describe the weighted score as measured density or queue length unless those quantities are independently measured.

### 3.4.4 Dual-Modal Emergency Verification

Define test cases for each supported emergency class: ambulance, firetruck, and police. Include positive and negative cases, including ordinary traffic audio, background noise, and cases where visual detection is absent or below threshold. Record the visual score, audio score, approach mapping, verification decision, and time at which the decision occurs.

A valid positive dual-modal case requires both scores to exceed 0.65 on the same approach. Evaluate false-positive and false-negative cases using a labelled test protocol. Do not claim that the dual-modal method outperforms single-modal alternatives unless a controlled comparison is conducted.

### 3.4.5 Signal Transition and Recovery Behaviour

Record the signal state before, during, and after an emergency-priority request. Verify that yellow and all-red intervals are maintained. Document how the system returns to its normal allocation after the emergency condition ends. Any claim about time compensation, recovery equilibrium, or reduction of secondary queues must be supported by a clearly defined implementation and measurements. If the current code does not implement a specific time-compensation mechanism, remove that mechanism from the manuscript rather than describing it as completed.

### 3.4.6 Environmental and Input-Quality Conditions

If environmental robustness is part of the study, define the actual lighting, weather, camera-angle, resolution, occlusion, and audio-noise conditions. Record the number of trials and ensure that each condition is represented in the evaluation data. Do not claim day/night/rain robustness unless those conditions were tested.

## 3.5 Performance Metrics and System Evaluation — define before testing

The evaluation should use metrics that correspond directly to the research questions:

1. **Vehicle detection:** precision, recall, F1-score, and mAP at a stated IoU threshold, with per-class reporting.
2. **Vehicle counting:** absolute count error or mean absolute error against manually annotated counts, if count accuracy is evaluated.
3. **Weighted allocation:** arithmetic verification of each approach score, green-time bounds, and total green-time budget.
4. **Emergency verification:** confusion matrix, precision, recall, F1-score, false-positive rate, and false-negative rate for the dual-modal decision.
5. **Response time:** elapsed time from the defined start of an emergency input to the system's verification decision or requested phase transition. State the start and end events precisely.
6. **Transition compliance:** measured duration of yellow and all-red intervals compared with the configured 3-second and 1-second values.
7. **Interface responsiveness:** report only if a reproducible measurement method is defined, such as processing throughput or interface update delay.

Traffic delay, queue reduction, throughput, emergency travel time, and recovery equilibrium require an explicit traffic-flow measurement setup and an appropriate baseline. Do not infer them from vehicle counts or green-time allocations alone.

---

# Items requiring author confirmation before manuscript finalization

| Item | Required confirmation |
|---|---|
| Research environment | Is the evaluated setup a software-only simulation, a miniature physical intersection, or both? The existing draft inconsistently describes these. |
| Video input | Which exact CCTV streams, webcams, and local videos were actually used in each experiment? |
| Audio dataset | Dataset source, class counts, recording conditions, train/validation/test split, random seed, and leakage controls. |
| Audio metrics | Original evidence for the reported validation Macro F1 of 92.17% and test accuracy of 86.70%, if the latter is to be included. |
| Audio model | Confirm the serialized artifact corresponds to Logistic Regression with (C=100), and provide the training manifest or notebook. |
| Vehicle model | Exact YOLO26n checkpoint, dataset, class distribution, split, training settings, and independent test metrics. |
| PCU/operational weights | Confirm whether the values are retained as simulator assumptions or recalibrated from an applicable local standard. |
| Timing algorithm | Confirm the exact bounded redistribution implementation and its behaviour when one or more approaches hit the 10- or 40-second bound. |
| Emergency transition | Confirm the actual next-phase selection and recovery behaviour from the current implementation. Do not retain a time-compensation claim unless it exists in code and was tested. |
| Research hypothesis | Confirm whether the primary inferential test concerns visual detection only or also includes audio verification and signal response. |
| References | Check every in-text citation against the final bibliography and the exact claim it supports. |

## Bibliography and citation cautions

- Reference [7] supports PCU estimation under the conditions of its own motorcycle-heavy signalized-intersection study. It does not independently validate every operational weight used by ARGUS.
- PKJI 2023 [28] is an appropriate official Indonesian reference to consult for road-capacity terminology and relevant vehicle categories. If its values are adopted, cite the exact section/table and explain why the selected conditions match the experiment.
- YAMNet documentation [27] should be cited for the pretrained audio feature extractor. It is model documentation, not a peer-reviewed experimental result for the project's classifier.
- YOLO26 should be cited using the relevant original model documentation or paper [17], while distinguishing the model architecture from the project's fine-tuned checkpoint.
- Research on multi-camera tracking [18] must not be used to imply that ARGUS itself performs persistent cross-camera identity tracking.
- Claims about emergency preemption and signal-transition safety should be supported by relevant transition literature [23, 24], but the existence of literature does not establish that ARGUS implements every mechanism described in those papers.
- Check the publication metadata and exact claim support for all references before final submission. Do not preserve a citation solely because it appeared in the original draft.

## Change log against the uploaded draft

- Replaced the abstract's unsupported claims about congestion reduction, immediate intersection clearance, and time compensation with an implementation-focused abstract.
- Reframed research questions and objectives so that they correspond to vehicle detection, weighted allocation, dual-modal verification, and evaluation.
- Narrowed the hypothesis to a measurable visual-detection outcome pending author confirmation of the statistical design.
- Corrected the description of the audio input from a standalone MP3/librosa heuristic to local-video soundtrack → YAMNet embeddings → Logistic Regression.
- Removed MFCC as an active GUI inference component; it may be reintroduced only if the implementation actually uses it.
- Added Police car consistently to the visual emergency classes.
- Replaced the ambiguous motorcycle weight of 0.25/0.33 with the current GUI operational value of 0.33, while explicitly warning that the whole table is not a universal PCU standard.
- Distinguished weighted traffic score from density, queue length, and image occupancy.
- Clarified the 120-second green allocation and the additional yellow/all-red intervals.
- Replaced statements that experiments “proved” superiority or improved traffic flow with a protocol for tests and metrics.
- Kept Chapter 4 and all result claims outside this draft.
