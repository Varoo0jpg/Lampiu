# ARGUS — Manuscript Revision Draft (I2ASPO, Maximum 13 Pages)

> **Note for Author — Revision basis and scope (not part of the manuscript)**
>
> This revision preserves the original manuscript's chapter order, section structure, research framing, and level of explanation wherever these remain consistent with the current ARGUS implementation. Changes are limited to correcting mismatches between the manuscript and the implemented system, aligning the number of research questions and objectives, and distinguishing literature-based PCU values from ARGUS configuration weights.
>
> The 13-page limit applies to the combined formatted manuscript from Chapter 1 through Chapter 5. The cover, table of contents, and bibliography are excluded according to the competition rules. This Markdown file does not have fixed pagination; verify the final count in the official I2ASPO template. Chapters 4–5 are not included in this working draft and must fit within the same 13-page body limit.
>
> **Note for Author — Page allocation:** Chapter 2 target: maximum 2 pages. Page allocation for the remaining chapters must be planned within the overall 13-page limit; no separate allocation should be treated as additional page allowance.

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

ARGUS allocates 120 seconds of green time across four approaches. Each receives a minimum of 10 seconds and a maximum of 40 seconds. After the minimum allocation, the remaining 80 seconds are distributed according to weighted approach scores using a bounded weighted water-filling procedure. When an approach reaches its maximum, it is removed from the active distribution and its remaining share is redistributed among the other approaches. If all approach scores are zero, the remaining time is divided equally. Integer rounding preserves the total allocation:

$$
\\sum_{i=1}^{4} G_i = 120, \\qquad 10 \\leq G_i \\leq 40
$$

Each approach is followed by 3 seconds of yellow and 1 second of all-red, giving a nominal normal rotation of:

$$
T_{\\mathrm{rotasi}} = 120 + 4(3 + 1) = 136\\,\\mathrm{s}
$$

This duration represents normal operation; emergency preemption or manual intervention may change the elapsed rotation time.

Emergency priority is requested only when visual emergency confidence and aggregate audio emergency score both strictly exceed 0.65 on the same approach:

$$
E_i = (V_i > 0.65) \\land (A_i > 0.65)
$$

A confirmed emergency request sets that approach as the next target. Yellow and all-red clearance intervals are retained. The thresholds are software settings, not field-calibrated safety parameters.

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

ARGUS integrates four-approach video input, YOLO26n vehicle detection, audio classification, weighted-demand estimation, signal-timing logic, and GUI monitoring. Visual feeds are processed per approach, while audio is obtained from the soundtrack of a user-selected local video. The system displays vehicle counts, weighted scores, audio scores, simulated signal states, and countdowns; periodic system-state records can be exported as CSV.

**Figure 3.1. ARGUS System Architecture**

[Insert the verified ARGUS system architecture diagram. Show the four visual inputs, local-video soundtrack input, visual and audio processing pipelines, weighted-demand calculation, signal allocation, emergency verification, GUI, and CSV logging. Do not depict live microphone input or physical signal control.]

*Source: ARGUS system design and implemented software.*

## 3.3 Software Implementation and Module Integration

### 3.3.1. Software Implementation

The visual module uses YOLO26n, Ultralytics, and OpenCV to detect eight vehicle classes, with confidence 0.15, NMS IoU 0.60, and five-frame median filtering. The audio module processes three-second mono segments at 16 kHz from a selected video's soundtrack; pretrained YAMNet generates 1,024-dimensional embeddings for Logistic Regression classification into ambulance, firetruck, police, and traffic.

For each approach, the detected class counts are multiplied by the corresponding operational weights:

\[
W_i=\sum_{c=1}^{8}N_{i,c}w_c
\]

where \(N_{i,c}\) is the detected count of class \(c\) on approach \(i\), and \(w_c\) is its operational weight. These simulator parameters are listed in **Table D.1 in Appendix D** and are not universal PCU standards. The controller allocates 120 seconds of green time across four approaches, with 10–40 seconds per approach; all-zero demand produces equal allocation. Yellow and all-red intervals are 3 and 1 seconds, respectively. The GUI displays detections, scores, signal states, and countdowns; periodic CSV snapshots record system states, not unique vehicles or cross-camera tracks.

### 3.3.2. Module Integration

The visual and audio pipelines operate independently and provide their outputs to the emergency-verification module. Priority is requested only when visual emergency confidence and aggregate audio emergency score both exceed 0.65 on the same approach. The selected approach receives the next eligible green phase after the configured yellow and all-red intervals. Otherwise, normal weighted-demand allocation continues. ARGUS remains a software simulation and is not connected to physical traffic-signal controllers.

**Figure 3.3. Dual-Modal Emergency Verification Flowchart**

[Insert the dual-modal emergency-verification flowchart showing same-approach matching, both threshold checks (>0.65), the verification decision, and priority at the next eligible green phase after yellow and all-red clearance. Include only branches confirmed by the final controller code.]

*Source: ARGUS emergency-verification logic. Verify the flowchart against the final implementation.*

## 3.4 Experimental Setup and Testing Scenarios

**Note for Author — Evaluation status (not part of the manuscript):** The following procedures define the evaluation to be performed. They are not experimental findings; results should be reported only after the corresponding tests have been run and recorded.

### 3.4.1 Experimental Environment

The simulation represents four approaches: North, East, South, and West. Record the video source assigned to each approach, computer and software specifications, YOLO26n checkpoint, and relevant GUI settings. For audio tests, also record the selected local video and its assigned approach. These details can be reported concisely in the text; a separate configuration table is unnecessary unless the final template or reviewer requires a consolidated parameter summary.

### 3.4.2 Heterogeneous Traffic and Green-Time Allocation

Test low-, medium-, and high-demand scenarios with different vehicle-class compositions. Calculate the expected approach scores using the operational weights in Table 3.1 and compare them with ARGUS outputs. Verify that each green duration is 10–40 seconds and that the four durations total 120 seconds. Include an all-zero-demand case to check equal allocation.

### 3.4.3 Dual-Modal Emergency Verification

Test ambulance, fire-engine, and police-vehicle cases under four conditions: both modalities exceed their thresholds; only the visual score exceeds its threshold; only the audio score exceeds its threshold; or neither does. Record the approach, both scores, and the verification decision. Confirmation requires both scores to be strictly greater than 0.65 on the same approach.

### 3.4.4 Signal Transition and Normal Rotation

Record the signal-phase sequence and interval durations during normal operation and emergency-priority requests. The nominal normal rotation comprises 120 seconds of green time, four 3-second yellow intervals, and four 1-second all-red intervals:

$
T_{\mathrm{rotation}}=120+4(3)+4(1)=136\text{ seconds}
$

For simultaneous emergency candidates, the software ranks approaches using the adjusted score:

$
S_i=W_i+0.01\,t_i
$

where $t_i$ is the elapsed time since approach $i$ last received green. This term affects candidate ranking only; it does not restore missed green time or demonstrate queue recovery. Verify that emergency requests retain the yellow and all-red clearance intervals.

## 3.5 Performance Metrics and System Evaluation

### 3.5.1 Vehicle-Detection Metrics

Evaluate YOLO26n using Precision, Recall, F1-score, and mean Average Precision (mAP). Determine true positives, false positives, and false negatives against labelled ground-truth annotations. The metrics are calculated as follows:

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

$$
F1=2\cdot\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
$$

$$
mAP=\frac{1}{C}\sum_{c=1}^{C}AP_c
$$

where TP is true positive, FP is false positive, FN is false negative, AP is average precision for a class, and C is the number of evaluated classes. Report per-class results and state the IoU convention; where available, include mAP50 and mAP50–95. The IoU thresholds used to calculate each mAP variant must be stated.

### 3.5.2 Traffic-Load and Green-Time Allocation Metrics

Compare ARGUS weighted scores with manually calculated scores from known vehicle counts and Table 3.1. For signal allocation, report each approach's green duration and verify the total allocation and per-approach limits:

$$
\sum_{i=1}^{4}G_i=120
$$

$$
10\leq G_i\leq40
$$

where $$G_i$$ is the green duration, in seconds, assigned to approach $$i$$. Queue length, average delay, throughput, and waiting-time improvement require a defined traffic-flow model or physical measurements and a baseline; they cannot be concluded from detection counts alone.

### 3.5.3 Emergency-Verification Metrics

Evaluate the dual-modal decision using a confusion matrix, Precision, Recall, F1-score, false-positive rate, and false-negative rate. Calculate Precision, Recall, and F1-score using the equations in Section 3.5.1. The additional error rates are:

$$
FPR=\frac{FP}{FP+TN}
$$

$$
FNR=\frac{FN}{FN+TP}
$$

where TN is true negative. Report the visual and audio scores separately and identify whether both thresholds were exceeded on the same approach. Define response time consistently, for example from the start of the relevant input segment to the verification decision.

### 3.5.4 Signal-Transition Compliance

Measure the yellow and all-red durations and compare them with the configured values of 3 seconds and 1 second. The absolute duration deviation can be calculated as:

$$
\Delta t=\left|t_{\mathrm{actual}}-t_{\mathrm{configured}}\right|
$$

A zero deviation indicates that the measured interval matches its configured duration. Record the phase sequence during normal operation and emergency-priority requests. Any claim about recovery time, restored traffic equilibrium, or reduced delay requires a separate recovery definition and supporting measurements.
---

# Note for Author — Checks Before Submission

**Note: Internal editing checklist; do not include this section in the submitted manuscript unless the competition template explicitly permits it.**

- Verify the source, definition, and year of the reported Yogyakarta vehicle-registration figure in the original reference [6].
- Confirm the PCU categories and values in Table 2.1 against the exact cited standard and reference. The simplified values must not be presented as universal.
- Confirm that Table 3.1 matches the operational weights in the current software.
- Verify the YOLO26n checkpoint, training dataset, data split, and independent test metrics before reporting model performance.
- Verify the audio dataset, class counts, train/validation/test split, and original training outputs before reporting any classifier metrics.
- Confirm that the test setup reflects the current local-video soundtrack selection behaviour.
- Do not claim physical traffic-light control, time compensation, reduced queue length, reduced emergency travel time, or field deployment unless those capabilities are implemented and directly evaluated.
- Check every in-text citation against the bibliography and the specific claim it supports.

**Page-count verification is required in the official I2ASPO Word template.** Chapters 1–5 combined must not exceed 13 pages, and Chapter 2 must not exceed 2 pages. Allocate the remaining space across Chapters 1, 3, 4, and 5; do not treat the earlier chapter allocations as valid page targets.
