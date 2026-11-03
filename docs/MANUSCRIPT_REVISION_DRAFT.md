# ARGUS — Manuscript Revision Draft (I2ASPO, Maximum 13 Pages)

> **Revision basis and scope**
>
> This revision preserves the original manuscript's chapter order, section structure, research framing, and level of explanation wherever these remain consistent with the current ARGUS implementation. Changes are limited to correcting mismatches between the manuscript and the implemented system, aligning the number of research questions and objectives, and distinguishing literature-based PCU values from ARGUS configuration weights.
>
> The 13-page limit applies to the combined formatted manuscript from Chapter 1 through Chapter 5. The cover, table of contents, and bibliography are excluded according to the competition rules. This Markdown file does not have fixed pagination; verify the final count in the official I2ASPO template. Chapters 4–5 are not included in this working draft and must fit within the same 13-page body limit.
>
> **Chapter 2 page target: maximum 2 pages.** Page allocation for the remaining chapters must be planned within the overall 13-page limit; no separate allocation should be treated as additional page allowance.

---

# ABSTRACT

Rapid urbanization in D.I. Yogyakarta has increased pressure on urban mobility, while fixed-time traffic controllers cannot directly adapt to changing traffic demand or differences among vehicle classes. Conventional signal control also provides limited support for identifying and prioritizing emergency vehicles. To address these issues, this study develops ARGUS (Adaptive Real-time Green-time Update System), a software-based traffic-signal simulation for a four-approach intersection. The visual subsystem uses a trained YOLO26n model to detect and classify eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Detected vehicle counts are processed using configured vehicle weights to calculate an approach-level traffic score and allocate bounded green-light durations. The emergency-verification subsystem combines visual emergency-vehicle detection with audio classification. Audio is extracted from the soundtrack of a user-selected local video, processed in three-second mono segments at 16 kHz, represented by pretrained YAMNet embeddings, and classified by a Logistic Regression model into ambulance, firetruck, police, and traffic categories. An emergency-priority request is made when the visual emergency confidence and aggregate audio emergency score exceed their configured thresholds on the same approach. The study describes the system architecture, software implementation, and procedures for evaluating vehicle detection, weighted timing allocation, and dual-modal emergency verification. ARGUS currently operates as a software simulation and is not connected to physical traffic-signal hardware; therefore, field-level reductions in congestion, vehicle delay, or emergency travel time are not claimed.

**Keywords:** Adaptive Traffic Control, Computer Vision, Emergency Vehicle Detection, Vehicle Weighting, Intelligent Transportation Systems.

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background of the Study

Rapid urbanization and escalating vehicular density pose critical challenges to modern urban mobility, directly undermining Sustainable Development Goal 11 (SDG 11: Sustainable Cities and Communities) by straining physical road infrastructure. Inefficient traffic signal control can further increase vehicle idling time and fuel consumption, which is relevant to SDG 13 (Climate Action). Moreover, conventional pre-timed traffic signals do not actively prioritize emergency response vehicles, potentially causing transit delays that compromise public safety and emergency preparedness under SDG 3 (Good Health and Well-being).

In developing urban centers such as the Special Region of Yogyakarta, Indonesia, rapid vehicular growth has increased pressure on a largely fixed road network. The original study background reports more than 3.5 million registered vehicles in 2025; this figure and its definition should remain supported by the corresponding local statistical reference [6]. Yogyakarta's traffic landscape is heterogeneous, with motorcycles, private passenger cars, public transportation, and freight vehicles sharing road space. Conventional pre-timed traffic controllers cannot respond directly to fluctuating traffic volumes. As a result, an approach with relatively low demand may receive green time while another approach experiences longer queues. Uniform vehicle counting also overlooks differences in the road-space requirements of vehicle classes, such as motorcycles and heavy trucks.

These limitations are intensified by the difficulty of identifying emergency vehicles in mixed traffic. Visual detection may be affected by occlusion, while audio detection may be affected by background noise or sounds that resemble sirens. Research on audio-based and audio-visual emergency-vehicle detection provides a basis for combining different sources of evidence [8–12]. A dual-modal approach can therefore be used as a verification condition before the simulation requests emergency priority.

To address these challenges, this study develops an Intelligent Transportation System framework named ARGUS (Adaptive Real-time Green-time Update System). ARGUS uses a trained YOLO26n object-detection model to classify vehicles from video sources representing four intersection approaches. The detected class counts are processed using vehicle weights to calculate approach-level traffic scores and allocate bounded green-light durations. In addition, the system combines visual emergency-vehicle detection with audio classification using pretrained YAMNet embeddings and a Logistic Regression classifier. Emergency priority is requested when both modalities meet their configured thresholds on the same approach.

ARGUS is currently a software-based traffic-signal simulation. It does not control physical traffic lights or establish field-level traffic improvements. This research focuses on implementing and evaluating the vehicle-detection, weighted signal-allocation, and dual-modal emergency-verification functions in controlled scenarios.

## 1.2 Problem Statement

This study addresses the following research questions:

1. How can a vehicle-weighting model based on the Passenger Car Unit (PCU) concept be applied to heterogeneous traffic to calculate approach-level traffic demand and allocate green time under normal operating conditions?
2. How does the signal-transition mechanism operate when the simulation grants emergency priority and subsequently returns to normal adaptive operation?
3. How does the integration of visual vehicle detection and audio classification perform in verifying emergency vehicles, and how does the resulting priority request affect green-time allocation among the four approaches in the software simulation?

## 1.3 Objectives of the Research

The objectives of this study correspond directly to the research questions:

1. To implement a PCU-concept-based vehicle-weighting calculation for heterogeneous traffic and use the resulting approach-level scores to allocate bounded green-light durations.
2. To implement and evaluate the signal transitions during emergency-priority requests and the return to normal adaptive operation.
3. To integrate visual vehicle detection and audio classification for emergency-vehicle verification, and evaluate its effect on simulated green-time allocation among the four approaches.

## 1.4 Significance of the Research

### 1.4.1 Theoretical Significance

1. **Advancement in Heterogeneous Traffic Modeling:** This study implements an algorithmic approach that accounts for multiple vehicle classes, including differences between motorcycles, passenger cars, buses, and trucks, rather than relying solely on uniform vehicle counts.
2. **Integration of Computer Vision and Emergency-Priority Logic:** This study combines visual object detection using YOLO26n with audio classification to establish a dual-modal emergency-verification condition.
3. **Framework for Software-Based Adaptive Traffic Control:** This study provides an implementation framework for combining multi-class vehicle detection, weighted traffic estimation, and adaptive signal timing in a software simulation.

### 1.4.2 Practical and Applied Significance

1. **Evaluation of Adaptive Signal Allocation:** The simulation provides a controlled environment for observing how weighted vehicle counts affect green-time allocation across four approaches.
2. **Evaluation of Emergency-Vehicle Verification:** The dual-modal mechanism provides a basis for examining emergency-verification decisions when visual and audio evidence are considered together.
3. **Basis for Further Development:** The software prototype can support subsequent research on local calibration, hardware integration, and field testing. Such deployment requires additional safety engineering and validation and is not claimed as an outcome of the current study.

## 1.5 Research Hypotheses

The primary detection hypothesis is formulated as follows:

1. **H0 (Null Hypothesis):** Under the defined test conditions, the YOLO26n model does not meet the predetermined vehicle-detection performance criterion.
2. **Ha (Alternative Hypothesis):** Under the defined test conditions, the YOLO26n model meets the predetermined vehicle-detection performance criterion.

The selected detection metric, acceptance criterion, test dataset, and evaluation procedure must be specified before statistical analysis. The hypothesis should not be interpreted as proof of reduced traffic delay or emergency travel time.

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Theoretical Framework

### 2.1.1 Traffic Flow Dynamics and Heterogeneous Characteristics

Traffic flow at signalized intersections is described by volume, speed, and density [5]. Yogyakarta traffic is heterogeneous and includes a substantial proportion of motorcycles [6]. Because vehicle classes occupy different amounts of road space, vehicle counts alone may not represent traffic demand adequately [1]. The Passenger Car Unit (PCU) concept expresses mixed vehicle classes as equivalent passenger-car units. PCU values vary with vehicle and road conditions; the original simplified reference values are retained in Table 2.1 [7, 16, 22, 28].

**Table 2.1. Passenger Car Unit (PCU) equivalent factors for heterogeneous traffic**

| Vehicle classification | PCU value | Reference basis |
|---|---:|---|
| Motorcycle (MC) | 0.25 / 0.33 | MKJI / Sugiarto et al. [7] (simplified) |
| Light Vehicle / Passenger Car (LV) | 1.00 | Passenger-car baseline [7] |
| Medium Heavy Vehicle / Bus (MHV) | 2.00 | MKJI / Sugiarto et al. [7] (simplified) |
| Heavy Vehicle / Large Truck (HV) | 2.00 | MKJI / Sugiarto et al. [7] (simplified) |

A study in motorcycle-dominated Banda Aceh traffic reported values of 0.24 for motorcycles and 1.00 for passenger cars, illustrating that PCU values are context-dependent [7]. Table 2.1 is a theoretical reference, not a direct mapping of all ARGUS detector classes. ARGUS uses separate operational weights, specified in Chapter 3.

The weighted approach score is:

$$
W_i=\sum_{c=1}^{C}N_{i,c}w_c
$$

where $$N_{i,c}$$ is the detected count of class $$c$$ on approach $$i$$ and $$w_c$$ is its assigned weight. This score estimates weighted demand; it does not directly measure queue length or density [3, 4].

### 2.1.2 Computer Vision

Computer vision detects and classifies road users from camera images and is widely applied in intelligent traffic systems [5, 20]. ARGUS uses a trained YOLO26n model to detect eight vehicle classes. Detection may be affected by lighting, camera angle, image quality, and occlusion; therefore, performance must be evaluated against labelled test data rather than confidence scores alone [17].

### 2.1.3 Acoustic Signals

Sirens provide an additional cue for emergency-vehicle identification, complementing visual detection when vehicles are occluded. Previous studies have investigated acoustic and audio-visual emergency-vehicle detection [8–12]. ARGUS uses pretrained YAMNet embeddings and a Logistic Regression classifier to classify audio as ambulance, firetruck, police, or traffic [27]. Its emergency audio score combines the three emergency-class scores. Audio is analysed from the soundtrack of a user-selected local video in three-second mono segments at 16 kHz.

### 2.1.4 Control Algorithm

Fixed-time control follows predetermined signal plans, whereas adaptive control adjusts timing to observed demand [2–5, 14]. Emergency preemption requires safe phase transitions rather than an immediate right-of-way change [23, 24].

ARGUS allocates 120 seconds of green time across four approaches. Each receives at least 10 seconds, with a maximum of 40 seconds; the remaining time is distributed according to weighted approach scores using a water-filling procedure. Capped shares are redistributed, and zero-demand cases receive equal allocation:

$$
\sum_{i=1}^{4}G_i=120,\qquad 10\leq G_i\leq40
$$

Each approach is followed by 3 seconds of yellow and 1 second of all-red, giving a nominal normal rotation of:

$$
T_{\mathrm{rotasi}}=120+4(3+1)=136\text{ seconds}
$$

Emergency priority is requested only when visual emergency confidence and the aggregate audio emergency score both exceed 0.65 on the same approach:

$$
E_i=(V_i>0.65)\land(A_i>0.65)
$$

The thresholds are software settings, and the signal retains its yellow and all-red clearance intervals.

# CHAPTER 3 — RESEARCH METHOD

## 3.1 System Requirements Analysis and System Characteristics

ARGUS (Adaptive Real-time Green-time Update System) is a software-based simulation of a four-approach traffic intersection. It combines vehicle detection using YOLO26n, weighted traffic-demand estimation, adaptive green-time allocation, and dual-modal emergency-vehicle verification. The system is intended for controlled software evaluation and is not connected to physical traffic-signal hardware [1–4].

### 3.1.1 System Requirements Analysis

The system uses video sources representing the North, East, South, and West approaches. Sources can be CCTV streams, webcams, or local video files selected through the launcher. The visual subsystem detects and counts eight vehicle classes. The resulting counts are processed using the operational weights in Table 3.1.

For emergency verification, the GUI analyses the soundtrack of a user-selected local video. The audio pipeline uses pretrained YAMNet embeddings and a Logistic Regression classifier. The software also requires a computer capable of running the GUI, video decoding, object detection, and audio inference. Physical signal controllers and IoT-based traffic-light actuation are outside the current implementation.

### 3.1.2 System Characteristics

The main system characteristics are:
1. Four-approach video input: North, East, South, and West.
2. Eight-class vehicle detection using a trained YOLO26n model.
3. Weighted approach scores used for bounded green-time allocation.
4. Audio classification from a selected local video's soundtrack.
5. Emergency verification requiring visual and audio scores above their thresholds on the same approach.
6. A GUI for displaying video, vehicle counts, weighted scores, audio scores, and simulated signal states, with research-data export in CSV format.

## 3.2 System Architecture and Algorithm Design

### 3.2.1 Input

Visual input consists of video streams or files assigned to the four approaches. Audio input is obtained from the soundtrack of a local video selected by the user. The soundtrack is decoded as stereo audio, divided into three-second segments, converted to mono, and resampled to 16 kHz before classification.

The quality of the inputs affects the resulting detections. Visual performance depends on camera position, lighting, resolution, and occlusion. Audio classification can be affected by background noise and soundtrack quality.

### 3.2.2 Process

The visual pipeline resizes video frames to 640 × 480 and processes them using YOLO26n. Detections are grouped by vehicle class and approach. A five-frame median filter is applied to the class counts to reduce short-term fluctuation. Each class count is multiplied by its operational weight, and the weighted values are summed to produce the approach score.

The signal-timing module uses one snapshot of the four approach scores for each normal rotation. It allocates 120 seconds of green time, with a minimum of 10 seconds and a maximum of 40 seconds per approach. The remaining time is distributed using weighted water-filling; when all approach scores are zero, the remaining time is divided equally.

For audio classification, YAMNet produces a 1,024-dimensional embedding for each three-second segment. Logistic Regression classifies the embedding into ambulance, firetruck, police, or traffic. The emergency audio score is calculated by adding the scores for the three emergency classes.

The emergency-verification module applies an AND condition: the visual emergency confidence and aggregate audio emergency score must both be greater than 0.65 and belong to the same approach. When confirmed, the approach is selected for the next green phase unless a manual override is active. The signal still passes through yellow and all-red clearance intervals.

### 3.2.3 Output

The GUI displays the four approach video panels, vehicle detections and class counts, weighted traffic scores, audio-classification scores, simulated red/yellow/green states, and countdown timers. It also provides source-selection and simulation controls. Research observations can be recorded and exported as CSV snapshots. The records describe system states at sampling times; they do not identify unique vehicles or provide cross-camera trajectories.

## 3.3 Software Implementation and Module Integration

### 3.3.1 Software Implementation

#### 3.3.1.1 YOLO26n-Based Vehicle Detection

Vehicle detection is implemented with the Ultralytics YOLO interface and OpenCV using the trained YOLO26n checkpoint. The model is configured for eight classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. The initial inference settings are confidence threshold 0.15, NMS IoU threshold 0.60, and image size 640; inference is configured for the CPU. Frames are resized to 640 × 480 before inference. The GUI's initial configuration has ROI filtering disabled; when enabled, the ROI must be configured for the relevant camera view.

Detections inside the selected counting area are counted by class. The maximum confidence among detected emergency classes (ambulance, fire engine, and police car) is retained as the visual emergency score for that approach. Vehicle counts are smoothed using a five-frame median filter before being passed to the traffic-load module.

#### 3.3.1.2 YAMNet-Based Audio Classification

The audio classifier is implemented in GUI/audio_siren.py and uses pretrained YAMNet with the stored Logistic Regression estimator best_audio_classifier_v2.joblib. The soundtrack of the selected local video is decoded by GUI/video_audio.py as stereo PCM at 44.1 kHz. The GUI processes three-second segments, averages the stereo channels to mono, resamples the segment to 16 kHz, and passes it to YAMNet. The resulting 1,024-dimensional embeddings are averaged and classified into ambulance, firetruck, police, and traffic.

The emergency audio score is calculated as:

$$
A_i=P(\text{ambulance})+P(\text{firetruck})+P(\text{police})
$$

The classifier scores are model outputs and are not claimed to be independently calibrated probabilities. The video playback can be muted while audio analysis continues. The analysed audio is associated with the approach assigned to the selected local video.

#### 3.3.1.3 PCU-Based Traffic-Load Estimation and Signal Control

The PCU categories in Table 2.1 provide the theoretical reference. ARGUS applies the following operational weights to its eight detector classes. These values are simulator parameters, not universal PCU standards.

**Table 3.1. Operational vehicle weights used in ARGUS**

| Detected vehicle class | Operational weight |
|---|---:|
| Ambulance | 1.00 |
| Bus | 2.00 |
| Car | 1.00 |
| Fire engine | 2.00 |
| Military vehicle | 2.00 |
| Motorcycle | 0.33 |
| Police car | 1.00 |
| Truck | 2.00 |

For each approach, the traffic score is calculated as:

$$
W_i=\sum_{c=1}^{8}N_{i,c}w_c
$$

The allocation begins with the minimum green time for each approach. The remaining time is distributed according to the active approach weights. If an approach reaches the maximum green time, it is capped and removed from the active distribution; the remaining allocation is then redistributed. If all active weights are zero, the remaining time is divided equally. Integer rounding uses the largest-remainder method so that the four green durations total exactly 120 seconds while remaining within the 10–40-second limits.

#### 3.3.1.4 Graphical User Interface and Research Data Logging

The GUI is developed using CustomTkinter and displays video feeds, vehicle counts, weighted traffic scores, audio scores, signal states, and countdown timers. It provides controls for video-source selection and simulation operation. During research recording, the system stores snapshots every second for each approach, including vehicle-class counts, total count, weighted score, signal phase, active approach, and data source. The records can be exported in CSV format and represent sampled system states rather than unique-vehicle tracking.

### 3.3.2 Module Integration

The visual pipeline supplies vehicle counts and visual emergency confidence for each approach. The traffic-load module applies the operational weights and calculates the four approach scores. Separately, the audio pipeline analyses the soundtrack of the selected local video and supplies the aggregate emergency audio score for its assigned approach.

The emergency-verification function confirms an approach only when both conditions are met:

$$
V_i>0.65\quad\text{and}\quad A_i>0.65
$$

The confirmed approach is requested as the next green phase unless a manual override is active. Signal transitions follow the sequence green → yellow → all-red → green; emergency priority does not bypass the clearance intervals. The GUI and detection processes operate through separate processing threads so that video and audio processing do not block the interface. All signal operations remain within the software simulation.

## 3.4 Experimental Setup and Testing Scenarios

The following procedures define the evaluation to be performed. They are not experimental findings; results should be reported only after the corresponding tests have been run and recorded.

### 3.4.1 Simulation Environment and Configuration

The test environment represents four approaches: North, East, South, and West. For each test, record the video source assigned to each approach, computer specifications, operating system, Python and package versions, YOLO26n checkpoint, and GUI settings. For audio tests, also record the selected local video and its assigned approach.

### 3.4.2 Heterogeneous Traffic and Green-Time Allocation

Prepare low-, medium-, and high-demand scenarios with known vehicle counts and different class compositions. Calculate the expected approach scores manually using Table 3.1, then compare them with the scores displayed by ARGUS. Record the four green durations and verify that each is between 10 and 40 seconds and that their sum is 120 seconds. Include an all-zero-demand scenario to verify equal allocation.

### 3.4.3 Dual-Modal Emergency-Vehicle Verification

Prepare positive and negative cases for ambulance, fire-engine, and police-vehicle detection. Include cases where both modalities indicate an emergency, only the visual modality exceeds its threshold, only the audio modality exceeds its threshold, and neither modality exceeds its threshold. Record the approach, visual confidence, audio emergency score, and verification decision. A positive decision requires both scores to be strictly greater than 0.65 on the same approach.

### 3.4.4 Signal Transition and Normal Rotation

Record the phase sequence and duration during normal operation and after an emergency-priority request. In a normal rotation, the configured duration is 120 seconds of green allocation plus four yellow intervals of 3 seconds and four all-red intervals of 1 second:

$$
T_{\mathrm{rotasi}}=120+4(3)+4(1)=136\text{ seconds}
$$

The code also calculates an adjusted approach score for selecting among simultaneously confirmed emergency candidates:

$$
S_i=W_i+0.01\,t_i
$$

where $$t_i$$ is the elapsed time since that approach last received green. This term is used in candidate ranking; it is not an algorithm that restores missed green time or proves that traffic queues return to equilibrium. Record the selected approach and verify that the yellow and all-red intervals are retained.

### 3.4.5 Environmental Robustness

If robustness is evaluated, test lighting variation, camera angle, partial occlusion, image quality, and audio background noise as separate conditions. Record the conditions and corresponding detection or classification outputs. Do not claim robustness for conditions that were not included in the test data.

## 3.5 Performance Metrics and System Evaluation

### 3.5.1 Vehicle-Detection Metrics

Evaluate YOLO26n using Precision, Recall, F1-score, and mean Average Precision (mAP). Determine true positives, false positives, and false negatives against labelled ground-truth annotations. Report per-class results and state the IoU convention; where available, include mAP50 and mAP50–95.

### 3.5.2 Traffic-Load and Green-Time Allocation Metrics

Compare ARGUS weighted scores with manually calculated scores from known vehicle counts and Table 3.1. For signal allocation, report each approach's green duration, the total green allocation, and whether the minimum and maximum limits are satisfied. Queue length, average delay, throughput, and waiting-time improvement require a defined traffic-flow model or physical measurements and a baseline; they cannot be concluded from detection counts alone.

### 3.5.3 Emergency-Verification Metrics

Evaluate the dual-modal decision using a confusion matrix, Precision, Recall, F1-score, false-positive rate, and false-negative rate. Report the visual and audio scores separately and identify whether both thresholds were exceeded on the same approach. Define response time consistently, for example from the start of the relevant input segment to the verification decision.

### 3.5.4 Signal-Transition Compliance

Measure the yellow and all-red durations and compare them with the configured values of 3 seconds and 1 second. Record the phase sequence during normal operation and emergency-priority requests. Any claim about recovery time, restored traffic equilibrium, or reduced delay requires a separate recovery definition and supporting measurements.

---

# Author Checks Before Submission

This checklist is an internal editing note and should be removed from the submitted manuscript unless the competition template permits it.

- Verify the source, definition, and year of the reported Yogyakarta vehicle-registration figure in the original reference [6].
- Confirm the PCU categories and values in Table 2.1 against the exact cited standard and reference. The simplified values must not be presented as universal.
- Confirm that Table 3.1 matches the operational weights in the current software.
- Verify the YOLO26n checkpoint, training dataset, data split, and independent test metrics before reporting model performance.
- Verify the audio dataset, class counts, train/validation/test split, and original training outputs before reporting any classifier metrics.
- Confirm that the test setup reflects the current local-video soundtrack selection behaviour.
- Do not claim physical traffic-light control, time compensation, reduced queue length, reduced emergency travel time, or field deployment unless those capabilities are implemented and directly evaluated.
- Check every in-text citation against the bibliography and the specific claim it supports.

**Page-count verification is required in the official I2ASPO Word template.** Chapters 1–5 combined must not exceed 13 pages, and Chapter 2 must not exceed 2 pages. Allocate the remaining space across Chapters 1, 3, 4, and 5; do not treat the earlier chapter allocations as valid page targets.
