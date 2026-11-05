# ARGUS: Adaptive Real-time Green-time Update System Using Multi-Camera YOLO Vehicle Detection and PCU-Based Weighting at a Four-Leg Intersection

**Compiled by:**
- Nabil Akbar Aprianto
- Alvaro Arta Putra
- Emir Javas Kawisworo
- Baihaqi Raihan Cahyadi
- Banyu Danadyaksa Karuna Gusti
- Almira Aznii Azizah

**SMA NEGERI 9 YOGYAKARTA**  
**2026/2027**

---

# ABSTRACT

Urban intersections experience changing traffic demand in which vehicle classes differ in their use of road space. Fixed-time signal plans do not directly respond to these observed variations, while emergency-vehicle prioritization requires reliable evidence before a signal phase is changed. This study presents ARGUS (Adaptive Real-time Green-time Update System), a software-based traffic-signal simulation for a four-approach intersection that combines multi-source vehicle detection, operational vehicle weighting, and dual-modal emergency-vehicle verification. The visual subsystem uses a trained YOLO26n model to detect and classify eight vehicle categories: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Detected vehicle counts are converted into an approach-level weighted traffic score using study-defined class weights. The controller distributes a total of 120 seconds of green time among the four approaches in proportion to these scores, subject to minimum and maximum green-time bounds.

The emergency-verification subsystem analyses the soundtrack of a user-selected local video. Audio is processed in three-second mono segments at 16 kHz, represented by 1,024-dimensional embeddings from pretrained YAMNet, and classified by a Logistic Regression model into ambulance, firetruck, police, and traffic categories. Emergency priority is requested only when the visual emergency-class confidence and the aggregate audio emergency score both exceed their configured thresholds on the same approach. Yellow and all-red intervals are retained during signal transitions. The study describes the system architecture, implementation, and an experimental protocol for evaluating detection, timing allocation, and emergency verification. Since ARGUS is currently a software simulation, the findings should not be interpreted as evidence of field effectiveness or measured traffic improvement.

**Keywords:** adaptive traffic signal; YOLO26n; passenger car unit; vehicle detection; YAMNet; emergency vehicle detection; dual-modal verification.

---

# CHAPTER 1
# Introduction

## 1.1. Background of the Study

Rapid urbanization and escalating vehicular density pose critical challenges to modern urban mobility, directly undermining Sustainable Development Goal 11 (SDG 11: Sustainable Cities and Communities) by straining physical road infrastructure. Inefficient traffic signal control further exacerbates idling times and greenhouse gas emissions, countering SDG 13 (Climate Action). Moreover, conventional pre-timed traffic signals do not provide adaptive emergency-vehicle priority, which may contribute to transit delays and challenges for emergency response under SDG 3 (Good Health and Well-being).

In developing urban centers such as the Special Region of Yogyakarta, Indonesia, traffic volume is substantial and the road network is shared by motorcycles, private passenger cars, public transit, and heavy freight. The BPS publication *Daerah Istimewa Yogyakarta Dalam Angka 2025* reports 3,328,178 motor vehicles for 2024. This is a 2024 figure published in the 2025 statistical yearbook, not a count of vehicles registered during 2025 [6]. Yogyakarta's traffic landscape is characterized by high heterogeneity, where unsegregated lanes are shared by different vehicle classes. Conventional pre-timed traffic controllers do not directly respond to fluctuating traffic volumes, potentially causing clear approaches to receive unnecessary green time while congested approaches experience prolonged queues. Furthermore, standard vehicle counting may neglect differences in the road-space and operating characteristics of vehicle classes.

These limitations are intensified when emergency-vehicle priority is unavailable or relies on only one detection modality. Vision-only detection may be affected by visual occlusion and image quality, while audio-only detection may be affected by ambient urban noise. Previous studies have investigated visual, acoustic, and audio-visual emergency-vehicle detection [8–13].

To address these challenges, this study proposes an Intelligent Transportation System (ITS) framework named ARGUS, using a computer-vision-driven traffic allocation model powered by a trained YOLO26n object-detection network. The system applies operational vehicle weights across eight detected vehicle classes to calculate an approach-level weighted traffic-demand score. This score is used to allocate bounded green durations; it is not a direct measurement of spatial occupancy, queue length, or traffic density. Additionally, a dual-modal verification framework combines visual recognition with YAMNet audio embeddings and a Logistic Regression classifier. The audio implementation does not use a separate MFCC feature-extraction stage. The study evaluates detection results, signal-transition timing, and green-time allocation in a four-approach software simulation. It does not implement a coordinated Green Wave across multiple physical intersections.

## 1.2. Problem Statement

1. How can a vehicle-weighting model based on the Passenger Car Unit (PCU) concept be applied to heterogeneous traffic to calculate approach-level traffic demand and allocate green time under normal operating conditions?
2. How does the signal-transition mechanism operate when the simulation grants emergency priority and subsequently returns to normal adaptive operation?
3. How does the integration of visual vehicle detection and audio classification perform in verifying emergency vehicles, and how does the resulting priority request affect green-time allocation among the four approaches in the software simulation?

## 1.3. Objectives of the Research

1. To implement a PCU-concept-based vehicle-weighting calculation for heterogeneous traffic and use the resulting approach-level scores to allocate bounded green-light durations.
2. To implement and evaluate the signal transitions during emergency-priority requests and the return to normal adaptive operation.
3. To integrate visual vehicle detection and audio classification for emergency-vehicle verification, and evaluate its effect on simulated green-time allocation among the four approaches.

## 1.4. Significance of the Research

### 1.4.1. Theoretical Significance

1. **Advancement in Heterogeneous Traffic Modeling:** Contributes to the field of Intelligent Transportation Systems (ITS) by establishing an algorithmic model that accounts for multi-class, non-homogeneous traffic flow (varying vehicle ratios like motorcycles vs. cars) rather than relying on uniform vehicle-count logic.
2. **Integration of Vision-Based Computer Vision and Emergency Preemption Logic:** Enriches the literature on computer-vision applications by combining object detection using YOLO with a priority-override algorithm for emergency vehicles.
3. **Framework for Low-Cost Computational Traffic Control Systems:** Provides an implementation example for executing vision-based traffic algorithms on accessible computing hardware. Its computational performance should be evaluated on the target hardware before making claims about real-world deployment.

### 1.4.2. Practical & Applied Significance

1. **Adaptive Green-Time Allocation:** Demonstrates how detected vehicle classes and their operational weights can be used to allocate green-light durations within configured limits. The current results do not establish a measured reduction in waiting time or an increase in intersection throughput.
2. **Potential for Further Infrastructure Integration:** Provides a software simulation that may serve as a basis for future integration research. ARGUS is not currently connected to physical municipal traffic-light infrastructure.
3. **Emergency-Vehicle Verification:** Demonstrates a dual-modal verification mechanism using visual detections and audio classification. The current implementation is limited to the four-approach simulation and does not establish an uninterrupted Green Wave corridor across multiple intersections.

## 1.5. Research Hypotheses

The primary detection hypothesis is formulated as follows:

1. **H0 (Null Hypothesis):** Under the defined test conditions, the YOLO26n model does not meet the predetermined vehicle-detection performance criterion.
2. **Ha (Alternative Hypothesis):** Under the defined test conditions, the YOLO26n model meets the predetermined vehicle-detection performance criterion.

The selected detection metric, acceptance criterion, test dataset, and evaluation procedure must be specified before statistical analysis. The hypothesis should not be interpreted as proof of reduced traffic delay or emergency travel time.

---

# CHAPTER 2
# Literature Review

## 2.1. Theoretical Framework

### 2.1.1. Traffic Flow Dynamics and Heterogeneous Characteristics

Traffic flow at signalized intersections is described by volume, speed, and density [5]. Yogyakarta traffic is heterogeneous and includes a substantial proportion of motorcycles [6]. Because vehicle classes occupy different amounts of road space, vehicle counts alone may not represent traffic demand adequately [1]. The Passenger Car Unit (PCU) concept expresses mixed vehicle classes as equivalent passenger-car units. PCU values vary with vehicle and road conditions; the original simplified reference values are retained in Table 2.1.

**Table 2.1. Passenger Car Unit (PCU) equivalent factors for heterogeneous traffic**

*Source: PCU references cited in the original manuscript. The exact MKJI edition supporting the simplified values is not yet verified.*

| Vehicle classification | PCU value | Reference standard stated in the original manuscript |
|---|---:|---|
| Motorcycle (MC) | 0.25 / 0.33 | MKJI / Sugiarto et al. [7] (simplified) |
| Light Vehicle / Passenger Car (LV) | 1.00 | Baseline standard |
| Medium Heavy Vehicle / Bus (MHV) | 2.00 | MKJI (simplified) |
| Heavy Vehicle / Large Truck (HV) | 2.00 | MKJI (simplified) |

A study in motorcycle-dominated Banda Aceh traffic reported values of 0.24 for motorcycles and 1.00 for passenger cars, illustrating that PCU values are context-dependent [7]. Table 2.1 is a theoretical reference, not a direct mapping of all ARGUS detector classes. ARGUS uses separate operational weights, specified in Chapter 3. **The MKJI edition/source supporting the simplified 0.25/0.33 and 2.00 entries is not included in the current reference list and must be added or verified before submission.**

The weighted approach score is:

\[
S_i = \sum_{c=1}^{C} n_{i,c}w_c
\]

where \(n_{i,c}\) is the detected count of class \(c\) on approach \(i\), and \(w_c\) is its assigned operational weight. This score estimates weighted demand; it does not directly measure queue length or density [3, 4].

### 2.1.2. Computer Vision

Computer vision detects and classifies road users from camera images and is widely applied in intelligent traffic systems [5, 20]. ARGUS uses a trained YOLO26n model to detect eight vehicle classes. Detection may be affected by lighting, camera angle, image quality, and occlusion; therefore, performance must be evaluated against labelled test data rather than confidence scores alone [19].

### 2.1.3. Acoustic Signals

Sirens provide an additional cue for emergency-vehicle identification, complementing visual detection when vehicles are occluded. Previous studies have investigated acoustic and audio-visual emergency-vehicle detection [8–13]. ARGUS uses pretrained YAMNet embeddings and a Logistic Regression classifier to classify audio as ambulance, firetruck, police, or traffic [27]. Its emergency audio score combines the three emergency-class scores. Audio is analysed from the soundtrack of a user-selected local video in three-second mono segments at 16 kHz.

### 2.1.4. Control Algorithm

Fixed-time control follows predetermined signal plans, whereas adaptive control adjusts timing to observed demand [2–5, 14]. Emergency preemption requires safe phase transitions rather than an immediate right-of-way change [23, 24].

ARGUS allocates 120 seconds of green time across four approaches. Each receives at least 10 seconds, with a maximum of 40 seconds; the remaining time is distributed according to weighted approach scores using a water-filling procedure. Capped shares are redistributed, and zero-demand cases receive equal allocation:

\[
\sum_{i=1}^{4} G_i = 120\ \text{seconds}, \qquad 10 \leq G_i \leq 40
\]

Each approach is followed by 3 seconds of yellow and 1 second of all-red. Therefore, the nominal full normal rotation is:

\[
T_{\text{rotation}} = 120 + 4(3) + 4(1) = 136\ \text{seconds}
\]

Emergency priority is requested only when visual emergency confidence and the aggregate audio emergency score both exceed 0.65 on the same approach:

\[
C_{\text{visual}} > 0.65 \quad \land \quad C_{\text{audio, emergency}} > 0.65
\]

The thresholds are software settings, and the signal retains its yellow and all-red clearance intervals.

---

# CHAPTER 3
# Research Method

## 3.1. System Requirements Analysis and System Characteristics

ARGUS (Adaptive Real-time Green-time Update System) is a computer-vision-based system developed to adjust simulated traffic-signal timing based on detected traffic conditions. Previous studies have explored computer-vision-based vehicle detection and adaptive traffic-signal control [1, 3]. ARGUS uses YOLO26n for vehicle detection from multiple CCTV feeds and webcams, operational vehicle weights based on the PCU concept to estimate weighted traffic demand [7], and an emergency-vehicle prioritization mechanism based on visual and audio classification.

### 3.1.1. System Requirements Analysis

ARGUS requires integration with supported Yogyakarta CCTV feeds as visual data sources and webcams or video files for experimental testing, alongside the YOLO26n model to detect and count vehicles across multiple approaches of a four-leg intersection. The detection results are processed using operational vehicle weights to estimate traffic demand and compare conditions across approaches [2, 7]. This analysis serves as a basis for adjusting green-light durations while maintaining the configured signal-transition constraints.

Furthermore, the system requires an emergency-vehicle verification mechanism using visual detection and audio classification, as well as real-time video processing capabilities, operational reliability, software compatibility, and an interface that presents traffic information clearly.

### 3.1.2. System Characteristics

ARGUS is characterized by computer-vision-based vehicle detection using YOLO26n and multi-camera monitoring to obtain traffic information from four simulated approaches [1, 2]. The system applies operational vehicle weights to estimate traffic demand based on vehicle categories, including motorcycles, passenger cars, buses, and trucks [7]. The resulting estimates support adaptive green-light timing adjustments according to observed conditions [2, 4]. Additionally, ARGUS incorporates an emergency-vehicle prioritization mechanism based on visual detection and audio classification, and uses supported CCTV feeds, webcams, or selected videos as visual data sources.

## 3.2. System Architecture and Algorithm Design

The ARGUS architecture integrates visual vehicle detection, audio classification, weighted traffic-demand estimation, signal-control logic, and graphical monitoring within a Python-based environment. Figure 3.1 illustrates the relationship between these components.

**Figure 3.1. ARGUS System Architecture**

[Insert the verified ARGUS system architecture diagram here. Show the four visual approaches, the local-video soundtrack audio path, YOLO26n, YAMNet and Logistic Regression, weighted-demand calculation, green-time allocation, dual-modal emergency verification, signal-phase simulation, GUI, and CSV logging.]

*Source: ARGUS system design and implemented software. The diagram must reflect the current code; do not depict a live microphone input or a physical traffic-signal controller.*

YOLO26n, implemented through the Ultralytics framework and OpenCV, detects eight vehicle classes—ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck—using a confidence threshold of 0.15 and an NMS IoU threshold of 0.60. A five-frame median filter stabilizes detection counts. Audio classification uses pretrained YAMNet embeddings and a Logistic Regression classifier to process three-second mono segments at 16 kHz from the soundtrack of a user-selected local video. The resulting emergency scores support dual-modal verification.

Detected vehicle counts are converted into operational weights to calculate a traffic-demand score for each approach. These scores govern allocation of a 120-second aggregate green-time budget, subject to a minimum of 10 seconds and a maximum of 40 seconds per approach; when all demand scores are zero, green time is divided equally. Signal transitions retain a three-second yellow interval and a one-second all-red clearance phase. Emergency priority is requested when visual emergency confidence and the aggregate audio emergency score both exceed 0.65 on the same approach. The CustomTkinter interface displays video feeds, detection statistics, signal phases, countdown timers, and audio-confidence metrics, while periodic system-state snapshots are exported in CSV format for quantitative evaluation.

## 3.3. Software Implementation and Module Integration

**Figure 3.2. ARGUS Software Module Integration Diagram**

[Insert a verified software-module integration diagram here. Show the four visual inputs and the local-video soundtrack audio input, YOLO26n and the YAMNet–Logistic Regression pipeline, weighted-demand estimation, green-time allocation, dual-modal emergency verification, signal-phase simulation, GUI, and CSV logging. Use arrows to distinguish data flow and control flow.]

*Source: ARGUS software architecture and implemented modules. Ensure that the diagram matches the final code and does not imply direct microphone input or physical signal-controller connectivity.*

### 3.3.1. Software Implementation

The visual detection module uses a trained YOLO26n model through the Ultralytics framework and OpenCV to identify eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. The recorded inference configuration uses a confidence threshold of 0.15, an NMS IoU threshold of 0.60, and a five-frame median filter to stabilize detection counts. The audio module processes three-second mono segments at 16 kHz, extracted from the soundtrack of a user-selected local video. Pretrained YAMNet generates 1,024-dimensional embeddings, which are classified by Logistic Regression into ambulance, firetruck, police, and traffic categories. The aggregate emergency score combines the three emergency categories.

For traffic-load estimation, detected vehicle counts are multiplied by their operational class weights to calculate an approach-level weighted-demand score:

\[
W_i=\sum_{c=1}^{8}N_{i,c}w_c
\]

where \(N_{i,c}\) is the detected count of vehicle class \(c\) on approach \(i\), and \(w_c\) is its corresponding operational weight. These weights are simulator parameters rather than universal PCU standards; the exact values used by ARGUS should be reported in **Table D.1 in Appendix D**. The controller distributes a 120-second aggregate green-time budget among the four approaches, with a minimum of 10 seconds and a maximum of 40 seconds per approach. It uses bounded weighted allocation, divides the time equally when all approach scores are zero, and retains the configured three-second yellow and one-second all-red intervals. The CustomTkinter GUI presents the four video panels, detection statistics, audio-confidence values, signal states, and countdown timers. Separate execution threads handle camera processing and audio analysis to maintain interface responsiveness, while periodic system-state snapshots are exported in CSV format. These snapshots do not represent unique-vehicle counts or cross-camera vehicle tracking.

### 3.3.2. Module Integration

The ARGUS modules are integrated through two primary processing pipelines: visual detection and audio classification. The visual pipeline processes camera feeds using YOLO26n to obtain vehicle classifications and counts, which are transferred to the traffic-load estimation module. Meanwhile, the audio pipeline processes video soundtracks using YAMNet and Logistic Regression to generate emergency-classification scores.

Both outputs are connected through a dual-modal verification mechanism. When visual and audio confidence scores exceed their respective 0.65 thresholds on the same approach, the system assigns emergency priority to that approach for the next eligible green-light phase. The transition maintains the yellow and all-red intervals before granting the next green signal. Otherwise, the simulation continues using traffic-load-based signal allocation. All processing results are displayed through the graphical interface, while separate execution threads handle camera processing and audio analysis to maintain interface responsiveness. Research data are recorded periodically and exported in CSV format. At the current development stage, ARGUS operates as a software-based traffic simulation and has not been connected to physical traffic-signal controllers.

**Figure 3.3. Dual-Modal Emergency Verification Flowchart**

[Insert the dual-modal emergency-verification flowchart here. Show visual emergency-candidate detection, audio-score evaluation, same-approach matching, both threshold checks (>0.65), the confirmed/not-confirmed decision, and the request for priority at the next eligible green phase after yellow and all-red clearance. Do not include a manual-override branch unless it is confirmed in the final controller code.]

*Source: ARGUS emergency-verification logic. Verify every decision branch against the final implementation.*

## 3.4. Experimental Setup and Testing Scenarios

### 3.4.1. Simulation Environment & Configuration

The test environment represents four approaches: North, East, South, and West. Record the video source assigned to each approach, computer specifications, operating system, Python and package versions, YOLO26n checkpoint, inference settings, audio configuration, signal parameters, and scenario duration. Training configuration and inference configuration must be identified separately.

**Table 3.1. Experimental Environment and System Configuration**

| Parameter | Recorded value / item to verify |
|---|---|
| Simulation approaches | North, East, South, West |
| Computer specifications and operating system | Record the machine and OS used for the reported experiment |
| Python and library versions | Record versions from the final experiment environment |
| YOLO26n checkpoint | Record the exact checkpoint filename and source |
| Inference confidence threshold | 0.15 (verify against final GUI configuration) |
| NMS IoU threshold | 0.60 (verify against final GUI configuration) |
| Inference image size and video-frame dimensions | Record the exact final values; do not substitute training image size |
| Detection smoothing | Five-frame median filter |
| Audio input | Soundtrack of a user-selected local video; record the file and assigned approach |
| Audio processing | Three-second mono segments at 16 kHz |
| Audio classifier | YAMNet embeddings (1,024 dimensions) and Logistic Regression |
| Green-time allocation | 120-second aggregate green budget; 10–40 seconds per approach |
| Signal transition | Three-second yellow and one-second all-red |
| Test scenario and duration | Record the actual scenario, source, and duration |

*Source: ARGUS configuration and experiment records. Values marked for verification must be checked against the final code and experiment logs before submission.*

### 3.4.2. Heterogeneous Traffic Density & PCU Weighting Protocol

Prepare low-, medium-, and high-demand scenarios with known vehicle counts and different class compositions. Calculate the expected approach scores manually using Table D.1, then compare them with the scores displayed by ARGUS. Record the four green durations and verify that each is between 10 and 40 seconds and that their sum is 120 seconds. Include an all-zero-demand scenario to verify equal allocation.

### 3.4.3. Dual-Modal Emergency Vehicle Preemption Protocol

Prepare positive and negative cases for ambulance, fire-engine, and police-vehicle detection. Include cases where both modalities indicate an emergency, only the visual modality exceeds its threshold, only the audio modality exceeds its threshold, and neither modality exceeds its threshold. Record the approach, visual confidence, audio emergency score, and verification decision. A positive decision requires both scores to be strictly greater than 0.65 on the same approach.

### 3.4.4. Transition Mechanism & Time-Compensation Validation

Record the phase sequence and duration during normal operation and after an emergency-priority request. In a normal rotation, the configured duration is 120 seconds of green allocation plus four yellow intervals of 3 seconds and four all-red intervals of 1 second, giving a nominal rotation of 136 seconds.

The code also calculates an adjusted approach score for selecting among simultaneously confirmed emergency candidates. The time-compensation term is based on the elapsed time since that approach last received green. This term is used in candidate ranking; it is not an algorithm that restores missed green time or proves that traffic queues return to equilibrium. Record the selected approach and verify that the yellow and all-red intervals are retained.

### 3.4.5. Environmental Robustness & Signal Degradation Testing

If robustness is evaluated, test lighting variation, camera angle, partial occlusion, image quality, and audio background noise as separate conditions. Record the conditions and corresponding detection or classification outputs. Do not claim robustness for conditions that were not included in the test data.

## 3.5. Performance Metrics and System Evaluation

### 3.5.1. Computer Vision & Object Detection Metrics

Evaluate YOLO26n using Precision, Recall, F1-score, and mean Average Precision (mAP). Determine true positives, false positives, and false negatives against labelled ground-truth annotations. The metrics are calculated as follows:

\[
\text{Precision} = \frac{TP}{TP+FP}
\]

\[
\text{Recall} = \frac{TP}{TP+FN}
\]

\[
F1 = 2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
\]

\[
mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c
\]

where TP is true positive, FP is false positive, FN is false negative, AP is average precision for a class, and C is the number of evaluated classes. Report per-class results and state the IoU convention; where available, include mAP50 and mAP50–95. The IoU thresholds used to calculate each mAP variant must be stated.

### 3.5.2. Traffic Flow & Queue Reduction Metrics

Compare ARGUS weighted scores with manually calculated scores from known vehicle counts and Table D.1. For signal allocation, report each approach's green duration and verify the total allocation and per-approach limits:

\[
\sum_{i=1}^{4}G_i=120\ \text{seconds}, \qquad 10\leq G_i\leq40
\]

where G_i is the green duration, in seconds, assigned to approach i. Queue length, average delay, throughput, and waiting-time improvement require a defined traffic-flow model or physical measurements and a baseline; they cannot be concluded from detection counts alone.

### 3.5.3. Preemption Latency & Dual-Modal Verification Accuracy

Evaluate the dual-modal decision using a confusion matrix, Precision, Recall, F1-score, false-positive rate, and false-negative rate. Calculate Precision, Recall, and F1-score using the equations in Section 3.5.1. The additional error rates are:

\[
FPR=\frac{FP}{FP+TN}, \qquad FNR=\frac{FN}{FN+TP}
\]

where TN is true negative. Report the visual and audio scores separately and identify whether both thresholds were exceeded on the same approach. Define response time consistently, for example from the start of the relevant input segment to the verification decision.

### 3.5.4. Signal Recovery & Transition Equilibrium

Measure the yellow and all-red durations and compare them with the configured values of 3 seconds and 1 second. The absolute duration deviation can be calculated as:

\[
E_t=|t_{\text{measured}}-t_{\text{configured}}|
\]

A zero deviation indicates that the measured interval matches its configured duration. Record the phase sequence during normal operation and emergency-priority requests. Any claim about recovery time, restored traffic equilibrium, or reduced delay requires a separate recovery definition and supporting measurements.

---

# CHAPTER 4
# Results and Discussion

## 4.1. Computer Vision Testing

The YOLO26n validation results are summarized in Table 4.1. The reported metrics should be interpreted in light of the dataset-annotation audit, which identified empty-label images requiring visual review. An empty label is not automatically an annotation error, but possible missing annotations can affect the interpretation of validation results.

**Table 4.1. YOLO26n Detection Performance by Class**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Ambulance | 97.6% | 91.1% | 97.8% | 89.0% |
| Bus | 88.2% | 71.4% | 85.3% | 66.5% |
| Car | 84.2% | 81.4% | 87.2% | 67.4% |
| Fire engine | 92.7% | 89.9% | 94.8% | 81.2% |
| Military | 71.8% | 90.0% | 88.1% | 65.6% |
| Motorcycle | 90.1% | 79.3% | 88.2% | 60.4% |
| Police car | 89.7% | 89.2% | 93.1% | 80.8% |
| Truck | 83.9% | 73.9% | 81.8% | 66.3% |
| **Overall** | **87.3%** | **83.3%** | **89.5%** | **72.1%** |

*Source: Reported YOLO26n validation results for the project. Record the exact run, dataset split, and evaluation configuration in the final manuscript.*

Figure 4.1 presents the normalized confusion matrix. It must be verified as originating from the same model run, validation split, and evaluation configuration as Table 4.1.

**Figure 4.1. Normalized Confusion Matrix**

[Insert `confusion_matrix_normalized.png` here after verifying that it corresponds to the same run and validation split as Table 4.1.]

*Source: YOLO26n validation output from the same experiment as Table 4.1.*

## 4.2. Traffic Control Evaluation

The traffic control evaluation was conducted to examine how ARGUS distributes green-light duration among the four approaches based on their weighted traffic scores. The system allocates a total of 120 seconds of green time, with a minimum duration of 10 seconds and a maximum duration of 40 seconds for each approach.

Based on the simulation results recorded in banyu4.csv, the green-time allocation for one normal cycle is presented in Table 4.2. **The 120 seconds refers to the aggregate green-time allocation, not the complete signal rotation. Including four 3-second yellow intervals and four 1-second all-red intervals, the nominal full normal rotation is 136 seconds.**

**Table 4.2. Green-Time Allocation for Each Approach**

[Retain the original green-time allocation table from the DOCX here. Its table cells were not available in the extracted text.]

The results show that the total green-time allocation reaches 120 seconds. Each approach receives a duration within the configured minimum and maximum limits. The East approach receives the longest green duration of 38 seconds, while the South approach receives the shortest duration of 22 seconds.

The simulation also records the signal transition sequence from green to yellow, followed by all-red before the next approach receives a green signal. The configured yellow and all-red intervals are 3 seconds and 1 second, respectively.

These results demonstrate that ARGUS can distribute green-light duration within the configured constraints and execute the normal signal transition sequence in the software simulation. However, these findings evaluate the timing allocation mechanism and do not directly demonstrate a reduction in vehicle queues, waiting time, or traffic congestion.

**Figure 4.2. Weighted-Demand Scores and Green-Time Allocation (Optional)**

[Insert a graph comparing weighted approach scores and green durations only if it is generated from the same verified experiment log as Table 4.2.]

*Source: The same verified experiment log used for Table 4.2.*

## 4.3. Emergency Vehicle Priority Testing

This section should report the actual visual detections, audio scores, dual-modal decisions, and response latency recorded during emergency-priority tests. The original manuscript does not provide a complete set of corresponding test results; therefore, no numerical performance claim is added here.

**Table 4.3. Dual-Modal Emergency Verification Results (if test data are available)**

[Include this table only after the test cases and ground truth are available. Suggested fields: test case, approach, visual emergency class and confidence, aggregate audio emergency score, same-approach match, verification decision, priority request, and measured latency. Do not populate missing values with estimates.]

*Source: ARGUS emergency-verification test logs. Include the test period and ground-truth procedure when available.*

## 4.4. System Recovery Evaluation

This section evaluates the signal sequence after an emergency-priority event and the return to normal traffic-load-based allocation. A complete recovery duration or traffic-equilibrium result should be reported only when the recovery interval has been explicitly defined and measured.

**Table 4.3. Emergency Signal Recovery and Phase-Transition Timing (Conditional)**

[Insert verified emergency-event and post-emergency timing records only when the complete test log and measurement definitions are available.]

**Figure 4.3. Emergency-Priority and Return-to-Normal Signal Timeline (Conditional)**

[Insert a signal-phase timeline based on the same verified event log as Table 4.3. Identify the emergency request, yellow interval, all-red clearance, emergency green phase, and return to normal allocation.]

*Source: ARGUS emergency-priority and recovery test logs. Do not infer missing timestamps or claim restored traffic equilibrium without supporting measurements.*

## 4.5. Overall System Analysis

Kelebihan, kekurangan, dan keterbatasan Lampiu.

[Complete this section after Sections 4.3 and 4.4 contain the actual test results. Do not claim measured queue reduction, delay reduction, throughput improvement, or multi-intersection Green Wave operation without supporting experiments.]

---

# CHAPTER 5
# Conclusion

[The original document contains drafting instructions rather than a completed conclusion. Complete this section after the emergency-priority and recovery test results have been finalized. The conclusion should summarize only findings reported in Chapter 4 and should not claim field effectiveness or traffic improvements that were not measured.]

---

# References

[1] Rahayu, R. (2023). Rancang Bangun Smart Traffic Light dengan Computer Vision sebagai Optimalisasi Pengaturan Lalu Lintas. S1 Teknik Informatika Thesis, STMIK Widya Cipta Dharma. Available: https://repository.wicida.ac.id/4999/2/2043909-S1-Jurnal.pdf

[2] Abbas, S. K., Khan, M. U. G., Zhu, J., Sarwar, R., Aljohani, N. R., Hameed, I. A., and Hassan, M. U. (2024). Vision Based Intelligent Traffic Light Management System Using Faster R-CNN. CAAI Transactions on Intelligence Technology, 9(4), 932–947. DOI: 10.1049/cit2.12309

[3] Osman, T., Psyche, S. S., Ferdous, J. M. S., and Zaman, H. U. (2017). Intelligent Traffic Management System for Cross Section of Roads Using Computer Vision. Proceedings of the 2017 IEEE 7th Annual Computing and Communication Workshop and Conference (CCWC), 1–7. DOI: 10.1109/CCWC.2017.7868350

[4] Teslyuk, V., Gado, I., Fylypchuk, B., and Koval, A. (2025). Methods and Tools for Automated Adaptive Traffic Light Control Based on Computer Vision. CEUR Workshop Proceedings, Vol. 4005. Available: https://ceur-ws.org/Vol-4005/paper4.pdf

[5] Serrano, Á., Conde, C., Rodríguez-Aragón, L. J., Montes, R., and Cabello, E. (2005). Computer Vision Application: Real Time Smart Traffic Light. In Computer Aided Systems Theory – EUROCAST 2005, Lecture Notes in Computer Science, Vol. 3643, pp. 525–530. DOI: 10.1007/11556985_68

[6] Badan Pusat Statistik Provinsi D.I. Yogyakarta. (2025). Daerah Istimewa Yogyakarta Dalam Angka 2025. BPS Provinsi D.I. Yogyakarta. [The 3,328,178 figure cited in the text refers to the 2024 vehicle count reported in this 2025 publication.]

[7] Sugiarto, S., Apriandy, F., Darma, Y., Saleh, S. M., Rusdi, M., and Miwa, T. (2021). Determining Passenger Car Equivalent (PCEs) for Pretimed Signalized Intersections with Severe Motorcycle Composition Using Bayesian Linear Regression. PLOS ONE, 16(9), e0256620. DOI: 10.1371/journal.pone.0256620

[8] Jayakumar, D., Krishnaiah, M., Kollem, S., Peddakrishna, S., Chandrasekhar, N., and Thirupathi, M. (2024). Emergency Vehicle Classification Using Combined Temporal and Spectral Audio Features with Machine Learning Algorithms. Electronics, 13(19), 3873. DOI: 10.3390/electronics13193873

[9] Tran, V.-T. and Tsai, W.-H., Acoustic-Based Emergency Vehicle Detection Using Convolutional Neural Networks, IEEE Access, 8, 75702-75713 (2020). DOI: 10.1109/ACCESS.2020.2988986

[10] Tran, V.-T. and Tsai, W.-H., Audio-Vision Emergency Vehicle Detection, IEEE Sensors Journal, 21(24), 27905-27917 (2021). DOI: 10.1109/JSEN.2021.3127893

[11] Kamelia, R. and Kusuma, H., Emergency Sound Classification and Visual Alert System for Enhanced Situational Awareness, 2024 International Conference on TVET Excellence and Development (ICTeD), IEEE, 213-218 (2024). DOI: 10.1109/ICTeD62334.2024.10844649

[12] Gabrielli, L., Migliorelli, L., Cantarini, M., Mancini, A. and Squartini, S., An Advanced Multimodal Driver-Assistance Prototype for Emergency-Vehicle Detection, Integrated Computer-Aided Engineering, 31(4), 381-399 (2024). DOI: 10.3233/ICA-240733

[13] Vikraman, B. P., Mahadevan, V., and Jabbar, R. F., “Revolutionizing Emergency Vehicle Response: A Smart Traffic Management Approach with Adaptive CNN and Hybrid Deep Learning,” International Journal of Information Technology, 17(3), 1793–1804 (2025). DOI: 10.1007/s41870-024-02088-2.

[14] Fahrunnisa, Z., Rahmadwati, R. and Setyawan, R. A., Adaptive Traffic Light Signal Control Using Fuzzy Logic Based on Real-Time Vehicle Detection from Video Surveillance, Jurnal Ilmiah Teknik Elektro Komputer dan Informatika (JITEKI), 10(2), 235-251 (2024). DOI: 10.26555/jiteki.v10i2.28712

[15] Majeed, A. I., Naeem, S. Q., Saeed, E. A. and Al-Shanoon, A., Real-Time Adaptive Traffic Signal Control with YOLOv10 and Image Processing, Al-Khwarizmi Engineering Journal, 21(4), 65-81 (2025). DOI: 10.22153/kej.2025.09.006

[16] Jithender, J. and Mehar, A., Modeling Passenger Car Unit and Delay at Signalized Intersections in Mixed Traffic Conditions, U.Porto Journal of Engineering, 9(3), 28-40 (2023). DOI: 10.24840/2183-6493_009-002_001508

[17] Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., and Kalfaoglu, M. E., “Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models,” arXiv preprint, arXiv:2606.03748 (2026). DOI: https://doi.org/10.48550/arXiv.2606.03748

[18] Huu, T. T. et al., “Proposing Smart System for Detecting and Monitoring Vehicle Using Multiobject Multicamera Tracking,” International Journal of Digital Multimedia Broadcasting, 2024, Article 6667738 (2024). DOI: https://doi.org/10.1155/2024/6667738

[19] He, J., Chen, H., Liu, B., Luo, S., and Liu, J., “Enhancing YOLO for Occluded Vehicle Detection with Grouped Orthogonal Attention and Dense Object Repulsion,” Scientific Reports, 14, Article 19650 (2024). DOI: https://doi.org/10.1038/s41598-024-70695-x

[20] Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., “You Only Look Once: Unified, Real-Time Object Detection,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779–788 (2016). DOI: https://doi.org/10.1109/CVPR.2016.91

[21] Hershey, S. et al., “CNN Architectures for Large-Scale Audio Classification,” Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 131–135 (2017). DOI: https://doi.org/10.1109/ICASSP.2017.7952132

[22] Hsu, T.-P., Huang, Y.-L., and Tsai, M. R., “Geometric Layout Dependent Passenger Car Equivalent of Motorcycle,” Journal of the Eastern Asia Society for Transportation Studies, 14, pp. 1888–1905 (2022). DOI: https://doi.org/10.11175/easts.14.1888

[23] Qin, X., and Khan, A. M., “Control Strategies of Traffic Signal Timing Transition for Emergency Vehicle Preemption,” Transportation Research Part C: Emerging Technologies, 25, pp. 1–17 (2012). DOI: https://doi.org/10.1016/j.trc.2012.04.004

[24] Yun, I., Best, M., and Park, B., “Evaluation of Transition Methods of the 170E and 2070 ATC Traffic Controllers after Emergency Vehicle Preemption,” Journal of Transportation Engineering, 134(10), pp. 423–431 (2008). DOI: https://doi.org/10.1061/(ASCE)0733-947X(2008)134:10(423)

[25] Noaeen, M. et al., “Reinforcement Learning in Urban Network Traffic Signal Control: A Systematic Literature Review,” Expert Systems with Applications, 199, Article 116830 (2022). DOI: https://doi.org/10.1016/j.eswa.2022.116830

[26] Özgün, S., and Öğüt, K. S., “Passenger Car Equivalent Estimation Methods at Urban Signalized Intersections: A Systematic Review,” Future Transportation, 6(1), Article 41 (2026). DOI: https://doi.org/10.3390/futuretransp6010041

[27] TensorFlow Models, “YAMNet: A Pre-trained Deep Net that Predicts 521 Audio Event Classes,” TensorFlow Models repository. https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

---

# APPENDICES

## Appendix A — System Implementation and GUI

- **Figure A.1. ARGUS GUI with Four Camera Panels** — [Insert a screenshot of the final GUI showing the four approaches.]
- **Figure A.2. Vehicle Detection Interface** — [Insert only if it demonstrates a distinct function not already clear in Figure A.1.]
- **Figure A.3. Audio Classification Display** — [Insert only if the final GUI displays this information.]
- **Figure A.4. Simulated Traffic-Signal States** — [Insert only if the signal-state display is not adequately visible in Figure A.1.]

*Source: Screenshots captured from the final ARGUS software version. Do not use screenshots from an outdated GUI build.*

## Appendix B — Vehicle Detection Examples

- **Figure B.1. Validation Batch 0 — Ground-Truth Labels** — [Insert `val_batch0_labels.jpg`.]
- **Figure B.2. Validation Batch 0 — Model Predictions** — [Insert `val_batch0_pred.jpg`.]
- **Figure B.3. Validation Batch 1 — Ground-Truth Labels** — [Insert `val_batch1_labels.jpg`.]
- **Figure B.4. Validation Batch 1 — Model Predictions** — [Insert `val_batch1_pred.jpg`.]
- **Figure B.5. Validation Batch 2 — Ground-Truth Labels** — [Insert `val_batch2_labels.jpg`.]
- **Figure B.6. Validation Batch 2 — Model Predictions** — [Insert `val_batch2_pred.jpg`.]

*Source: The same YOLO26n validation run used for Table 4.1. Keep each label/prediction pair from the same batch.*

## Appendix C — Model Training and Evaluation

- **Figure C.1. YOLO26n Training and Validation Curves** — [Insert `results.png`.]
- **Table C.1. YOLO26n Training Results by Epoch** — [Attach or summarize `results.csv` only if the complete file is available.]
- **Figure C.2. YOLO26n Box Precision–Recall Curve** — [Insert `BoxPR_curve.png`, if available.]
- **Figure C.3. YOLO26n Box Precision Curve** — [Insert `BoxP_curve.png`, if available.]
- **Figure C.4. YOLO26n Box Recall Curve** — [Insert `BoxR_curve.png`, if available.]
- **Figure C.5. YOLO26n Box F1 Curve** — [Insert `BoxF1_curve.png`, if available.]
- **Figure C.6. Normalized Confusion Matrix** — [Refer to Figure 4.1; do not duplicate the same image unless the appendix copy is required by the submission format.]
- **Figure C.7. Training Batch Examples** — [Insert selected `train_batch*.jpg` files only as training-process documentation, not as model-accuracy evidence.]

*Source: The same training run as the reported checkpoint. Verify all files belong to that run.*

## Appendix D — Traffic Allocation and Signal Timing

- **Table D.1. Operational Vehicle-Class Weights Used by ARGUS** — [Insert the exact eight class weights from the final controller code or project documentation. Do not infer missing values.]
- **Table D.2. Example Weighted-Demand Calculation** — [Insert a worked example based on verified vehicle counts and the weights in Table D.1.]
- **Table D.3. Example Green-Time Allocation** — [Insert a calculation or logged allocation supported by the experiment records.]
- **Figure D.1. Signal Phase Sequence** — [Insert a diagram of green → yellow → all-red → next green, using the configured durations.]
- **Table D.4. Signal Timing Test Records** — [Insert only if the corresponding logs are available.]

*Source: Final ARGUS controller configuration and verified test logs. The weights are simulator parameters, not universal PCU standards.*

## Appendix E — Dataset and Threshold Audit

- **Figure E.1. Dataset Contact Sheet** — [Insert `contact_sheet.jpg`.]
- **Figure E.2. Empty-Label Training Samples** — [Insert `dataset_audit/empty_train.jpg`.]
- **Figure E.3. Empty-Label Validation Samples** — [Insert `dataset_audit/empty_valid.jpg`.]
- **Table E.1. Threshold Sweep Summary** — [Insert or summarize `threshold_sweep/threshold_summary.csv` if available.]
- [Add class-specific contact sheets only when they support a specific annotation-audit discussion.]

*Source: Dataset audit artifacts. The previously recorded audit identified 1,128 empty-label images (923 train and 205 valid); the set is mixed and must not be treated as entirely correct or entirely incorrect without visual verification.*

## Appendix F — Frame Verification and Error Analysis

- **Figure F.1. Frame 0036 — Visual Verification** — [Insert `frame_0036.jpg`.]
- **Figure F.2. Frame 0048 — Visual Verification** — [Insert `frame_0048.jpg`.]
- **Figure F.3. Frame 0157 — Potential Police-Car False Positive** — [Insert `frame_0157.jpg`.]
- **Figure F.4. Frame 0172 — Ambulance Detection Example** — [Insert `frame_0172.jpg`.]
- **Figure F.5. Frame 0175 — Visual Verification** — [Insert `frame_0175.jpg`.]
- **Figure F.6. Frame 0278 — Visual Verification** — [Insert `frame_0278.jpg`.]
- [Add threshold-audit examples from `val_conf025` or `val_conf001` only when their thresholds and evaluation context are stated.]

*Source: Project frame-verification and error-audit artifacts. Label each example as a correct detection, false positive, false negative, or ambiguous case according to the actual inspection.*

## Appendix G — Audio Evaluation

[Complete this appendix only after audio evaluation data and ground truth are available.]

- **Table G.1. Audio Dataset Description and Class Distribution** — [Insert verified dataset details.]
- **Figure G.1. Audio Classification Confusion Matrix** — [Insert only if an evaluation was conducted.]
- **Table G.2. Audio Classification Metrics by Class** — [Insert only verified metrics.]
- **Figure G.2. Audio Test Examples or Spectrograms** — [Insert only when generated from documented test samples.]

*Source: Audio evaluation dataset and test records. Do not invent evaluation results.*

## Appendix H — System Logs and Testing Records

- **Table H.1. ARGUS System Log Fields** — [Insert the relevant, sanitized log-field description.]
- **Table H.2. Vehicle Detection Log Fields** — [Insert the relevant, sanitized log-field description.]
- **Table H.3. Audio Classification Log Fields** — [Insert only if the audio log exists.]
- **Table H.4. Emergency Verification Log Fields** — [Insert only if the verification log exists.]
- **Table H.5. Signal-State Log Fields** — [Insert the relevant, sanitized log-field description.]

*Source: ARGUS CSV/log exports. One-second records are periodic system-state snapshots, not unique-vehicle counts or cross-camera tracking records.*

---

# AUDIT CHANGE LOG — COMPARISON WITH THE ORIGINAL DOCX

The original chapter order, section order, research questions, objectives, main terminology, and reference-list order have been retained. The following are the substantive changes made in this Markdown audit copy.

| Original location | Change made | Reason |
|---|---|---|
| 3.2 | Consolidated the architecture diagram and system summary into one subchapter; removed the separate Input and Process subsections | Reduces repetition and preserves the main architecture, input, processing, timing, emergency-verification, GUI, and logging information |
| 3.3 | Consolidated implementation details into Software Implementation and Module Integration, added Figure 3.2 for module integration, and retained Figure 3.3 for the dual-modal verification flowchart | Reduces nested subsections while retaining the implementation and data-flow explanation |
| Tables and figures | Retained Table 3.1 for experimental configuration, Table D.1 for operational vehicle weights, Tables 4.1–4.3 for detection, allocation, and conditional recovery results, and Figures 3.1–3.3 and 4.1–4.3 for architecture, integration, verification, and result visuals | Applies sequential numbering and descriptive captions to the figure/table placeholders without fabricating missing data or images |
| Abstract | Removed the statement that the manuscript has no completed experimental results; retained the original system description and clarified that it is a software simulation | The project has reported model-validation and software-timing results, but these do not establish field effectiveness |
| 1.1 Background | Replaced “exceeded 3.5 million registered units in 2025” with the BPS figure of 3,328,178 for 2024, published in the 2025 yearbook | Corrects the year and figure |
| 1.1 Background | Replaced claims that ARGUS computes spatial occupancy and uses MFCC with weighted-demand scoring and YAMNet plus Logistic Regression | Aligns the description with the implemented method |
| 1.1 Background and 1.4.2 | Removed unsupported claims of a Green Wave corridor, drastic delay reduction, and measured throughput improvement | The implementation is a four-approach software simulation, not coordinated multi-intersection control |
| 2.1.1 PCU table | Retained the original table and its values; added a note that the MKJI source supporting the simplified values is absent from the reference list | Preserves the original theoretical framework while making a citation gap explicit |
| 2.1.1 | Restored the weighted-demand equation and clarified that it does not directly measure queue length or density | Prevents the weighted score from being misrepresented as a physical traffic measurement |
| 2.1.2 | Corrected the occlusion citation to reference [19] | Reference [17] is the YOLO26 paper, not the occlusion study |
| 2.1.3 and 3.3.1.2 | Removed MFCC from the audio implementation description and corrected the subsection heading | The described implementation uses YAMNet embeddings and Logistic Regression |
| 2.1.3 and References | Added YAMNet's TensorFlow Models reference as [27] | The original manuscript cited [27] although its bibliography ended at [26] |
| 2.1.4, 3.3.1.3 and 3.4.4 | Clarified that 120 seconds is the aggregate green allocation; full normal rotation is 136 seconds | Adds four 3-second yellow intervals and four 1-second all-red intervals |
| 3.2.1 | Replaced the separate MP3/librosa heuristic description with soundtrack extraction, 16 kHz mono segments, YAMNet embeddings, and Logistic Regression | Aligns audio input and processing with the implemented pipeline |
| 3.3.1.4 | Qualified the Auto/Manual selector claim | The GUI selector exists, but the controller does not consume the mode variable |
| 4.1 | Added the reported overall and per-class YOLO26n validation metrics | Fills the original result placeholder with results already reported for the project |
| 4.2 | Retained the original East 38-second and South 22-second result description; clarified the meaning of the 120-second allocation | Corrects terminology without replacing the original experiment with a different dataset or CSV analysis |
| 4.3–4.5 and Chapter 5 | Kept the original section sequence but marked missing result/conclusion content explicitly instead of inventing results | The original document contains headings/topic notes and drafting instructions, not complete emergency/recovery results |
| References | Preserved original references [1]–[26] in their original order and added [27] for YAMNet | Avoids wholesale renumbering and repairs the missing YAMNet entry |

## ITEMS STILL REQUIRING SOURCE CONFIRMATION

1. Add the exact MKJI edition/publication supporting the 0.25/0.33 motorcycle and 2.00 bus/heavy-vehicle PCU values in Table 2.1.
2. Verify the exact YOLO26n run, dataset split, and confusion-matrix image used for Table 4.1 and Figure 4.1.
3. Add the actual emergency-priority test results before completing Section 4.3.
4. Add a defined and fully measured recovery result before completing Section 4.4.
5. Complete Chapter 5 only after the final Chapter 4 results are settled.
6. Verify all Table 3.1 configuration values and the exact eight operational weights in Table D.1 against the final code and experiment logs.
7. Insert Figure 3.1 and Figure 3.2 only after confirming that both diagrams match the final implementation.
8. Populate conditional Chapter 4 and appendix placeholders only when the corresponding original assets, logs, and evaluation data are available.

**Scope note:** This revision reorganizes visual/table placeholders according to the supplied placement guide. It does not assert that all referenced image files, tables, logs, or test results are already available. The original Chapter 4.2 allocation result (East 38 seconds; South 22 seconds) remains unchanged. CSV contents were not independently audited to replace that result.
