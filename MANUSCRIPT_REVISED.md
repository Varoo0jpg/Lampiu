# ARGUS: Adaptive Real-time Green-time Update System Using Multi-Camera YOLO Vehicle Detection and PCU-Based Weighting at a Four-Leg Intersection

**Research manuscript — revised Markdown draft**  
**Institution:** SMA Negeri 9 Yogyakarta  
**Academic year:** 2026/2027

**Compiled by**
- Nabil Akbar Aprianto
- Alvaro Arta Putra
- Emir Javas Kawisworo
- Baihaqi Raihan Cahyadi
- Banyu Danadyaksa Karuna Gusti
- Almira Aznii Azizah

> **Editorial note.** This revision aligns the manuscript with the described ARGUS/Lampiu software implementation. It does not claim physical traffic-controller deployment, multi-intersection coordination, measured traffic-delay reduction, queue-length measurement, or direct live-audio capture from all CCTV feeds. Figures that could not be embedded from the available source are represented by explicit placeholders. Replace each placeholder with the corresponding original project image before submission.

---

## ABSTRACT

Urban intersections experience changing traffic demand, while vehicle classes differ in their operational contribution to mixed traffic. Fixed-time signal plans do not directly respond to observed demand, and emergency-vehicle prioritization requires evidence before a signal phase is changed. This study presents ARGUS (Adaptive Real-time Green-time Update System), a software-based traffic-signal simulation for a four-approach intersection. ARGUS combines multi-camera vehicle detection, study-defined vehicle weighting, and dual-modal emergency-vehicle verification.

The visual subsystem uses a trained YOLO26n model to detect eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Detected counts are converted into an approach-level weighted demand score. The controller distributes an aggregate green-time budget of 120 seconds among the four approaches, subject to minimum and maximum allocations of 10 and 40 seconds. Each normal approach phase is followed by 3 seconds of yellow and 1 second of all-red clearance, giving a nominal full normal rotation of 136 seconds.

The emergency-verification subsystem analyses audio from the soundtrack of a user-selected local video. Audio is processed in three-second mono segments at 16 kHz using pretrained YAMNet embeddings and a Logistic Regression classifier. Emergency priority is requested when visual emergency-vehicle confidence and the aggregate audio emergency score both exceed 0.65 on the same approach. The software retains yellow and all-red clearance intervals during transitions.

The reported model evaluation gives overall precision of 87.3%, recall of 83.3%, mAP@0.5 of 89.5%, and mAP@0.5:0.95 of 72.1%. A software timing test recorded an emergency request-to-emergency-green interval of 3.062 seconds in the observed event. These results describe model validation and software-simulation behaviour; they do not establish field effectiveness or a reduction in real-world travel time.

**Keywords:** adaptive traffic signal; YOLO26n; vehicle detection; passenger car unit; weighted traffic demand; YAMNet; emergency-vehicle verification; software simulation.

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background of the Study

Traffic demand at urban intersections changes over time and consists of different vehicle classes. Motorcycles, passenger cars, buses, and trucks do not have identical operating characteristics or road-space requirements. Consequently, a simple vehicle count may not represent the relative demand of a mixed-traffic approach. The Passenger Car Unit (PCU) concept provides one way to express heterogeneous vehicle classes relative to passenger-car units, although its values depend on local traffic and road conditions [1–3].

In the Special Region of Yogyakarta, road traffic includes a substantial mixture of motorcycles and other vehicle types. According to *Daerah Istimewa Yogyakarta Dalam Angka 2025*, the recorded number of motor vehicles in DIY in 2024 was 3,328,178 [4]. This figure provides regional context; it should not be interpreted as a 2025 vehicle-registration count.

Conventional fixed-time traffic signals use predetermined timing plans. Such plans may not reflect short-term differences in demand between approaches. Computer-vision-based traffic monitoring offers a way to estimate vehicle counts from camera images and use the resulting information in adaptive signal-control research [5–8]. However, detection performance can be affected by lighting, camera position, image quality, and occlusion [9].

Emergency vehicles introduce an additional control requirement. A siren can provide an acoustic cue when the corresponding vehicle is partly obscured or difficult to identify visually. Previous research has investigated acoustic and audio-visual emergency-vehicle detection [10–13]. A combined visual and audio decision can therefore be used as a software verification mechanism, provided that its limitations and input sources are clearly stated.

This study develops ARGUS (Adaptive Real-time Green-time Update System), a software-based simulation for a four-approach intersection. It uses YOLO26n to detect eight vehicle classes, applies study-defined operational weights to estimate weighted demand, and distributes a 120-second aggregate green-time budget across four approaches. It also combines visual emergency-vehicle detections with audio classification from the soundtrack of a user-selected local video. ARGUS is evaluated as a software system; it is not connected to physical traffic-signal controllers.

## 1.2 Problem Statement

This study addresses the following questions:

1. How can detected vehicle counts and study-defined class weights be used to calculate an approach-level weighted demand score and allocate bounded green time?
2. How does the software maintain signal-clearance intervals when an emergency-priority request is processed?
3. How do the visual and audio subsystems contribute to emergency-vehicle verification in the four-approach simulation?
4. What model-validation and software-timing results can be reported from the available experiments, and what conclusions remain unsupported by the current evidence?

## 1.3 Objectives of the Research

The objectives are:

1. To implement vehicle detection and classification using the trained YOLO26n model.
2. To calculate a weighted demand score for each approach using detected vehicle counts and operational class weights.
3. To distribute the aggregate green-time budget while observing the configured minimum and maximum green-time bounds.
4. To integrate visual emergency-vehicle detection with audio classification from a selected local video.
5. To evaluate detection metrics, example green-time allocations, and recorded signal-transition timing in the software simulation.

## 1.4 Significance of the Research

### 1.4.1 Theoretical Significance

The study provides an implementation example of combining multi-class vehicle detection with weighted approach-demand estimation. It also documents a dual-modal emergency-verification workflow and the use of clearance intervals during software-based signal transitions. The approach score is treated as a weighted demand indicator, not as a direct measurement of queue length, density, or delay.

### 1.4.2 Practical and Applied Significance

ARGUS provides a graphical environment for observing four video feeds, vehicle detections, approach-level demand scores, signal states, countdowns, and audio-confidence information. It can support controlled demonstrations and software experiments. Its current implementation does not establish that the system can be retrofitted to municipal traffic infrastructure or that it reduces real-world congestion or emergency travel time.

## 1.5 Research Hypotheses

The primary quantitative hypothesis concerns the vehicle-detection model:

- **H0 (null hypothesis):** Under the defined evaluation conditions, YOLO26n does not meet the predetermined vehicle-detection performance criterion.
- **Ha (alternative hypothesis):** Under the defined evaluation conditions, YOLO26n meets the predetermined vehicle-detection performance criterion.

The selected metric, acceptance threshold, evaluation dataset, and test procedure must be specified before formal hypothesis testing. Model-detection performance alone does not prove a reduction in traffic delay or emergency travel time.

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Theoretical Framework

### 2.1.1 Traffic Flow and Heterogeneous Vehicle Characteristics

Traffic flow is commonly described using variables such as volume, speed, and density [5]. In mixed traffic, vehicle classes have different operating characteristics. The PCU concept expresses the effect of a vehicle class relative to a passenger car. PCU values are context-dependent and may vary with traffic composition, road geometry, and operating conditions [1–3].

**Table 2.1. Selected PCU reference values reported in a motorcycle-dominated traffic study**

| Vehicle class | PCU value | Interpretation |
|---|---:|---|
| Motorcycle | 0.24 | Context-specific value reported for the studied traffic conditions |
| Passenger car | 1.00 | Reference passenger-car unit |

*Source: Sugiarto et al. [1]. These values are a literature reference, not a complete mapping for every ARGUS detector class. Bus and truck values should not be inferred from this table.*

ARGUS uses its own operational class weights in the controller. These implementation weights are conceptually separate from the theoretical PCU values in Table 2.1.

The weighted demand score for approach i is:

\[
S_i = \sum_{c=1}^{C} n_{i,c}w_c
\]

where:

- S_i is the weighted demand score for approach i;
- n_{i,c} is the detected count of vehicle class c on approach i;
- w_c is the operational weight assigned to class c;
- C is the number of vehicle classes.

This score estimates weighted demand from detected counts. It does not directly measure queue length, spatial occupancy, traffic density, or vehicle delay.

### 2.1.2 Computer Vision for Vehicle Detection

Computer vision enables the detection and classification of road users from camera images and has been applied in intelligent traffic systems [5–8]. The YOLO family is designed for object detection, while newer versions continue to develop real-time detection architectures [14,15]. Occlusion and difficult visual conditions remain important sources of detection error [9].

ARGUS uses a trained YOLO26n model with eight output classes:

1. Ambulance
2. Bus
3. Car
4. Fire engine
5. Military
6. Motorcycle
7. Police car
8. Truck

The model's performance is assessed using precision, recall, mAP@0.5, and mAP@0.5:0.95. Confidence values displayed by the GUI should not be interpreted as a substitute for evaluation against labelled data.

### 2.1.3 Acoustic Signals and Emergency-Vehicle Verification

Acoustic emergency-vehicle detection has been studied using machine-learning methods and audio features [10,11]. Audio-visual approaches combine acoustic information with visual evidence [12,13]. YAMNet is a pretrained audio-event model that produces 1,024-dimensional embeddings and class scores for audio events [16].

ARGUS uses YAMNet embeddings with a Logistic Regression classifier. The classifier outputs four categories: ambulance, firetruck, police, and traffic. The aggregate emergency audio score combines the three emergency-category scores. Audio is obtained from the soundtrack of a user-selected local video, converted to mono, sampled at 16 kHz, and analysed in three-second segments. This is not equivalent to direct live-audio capture from every CCTV source.

### 2.1.4 Adaptive Signal Allocation and Emergency Preemption

Adaptive traffic-signal methods adjust signal timing in response to estimated traffic demand [6–8,17]. Emergency-vehicle preemption requires a safe transition between signal phases; it should not be implemented as an instantaneous change that bypasses clearance intervals [18,19].

ARGUS distributes a total green-time budget of 120 seconds among four approaches. The configured allocation bounds are 10 seconds minimum and 40 seconds maximum per approach. A capped-share redistribution (water-filling) procedure redistributes the remaining allocation when an approach reaches its bound. When the total detected demand is zero, the software uses equal allocation.

The normal green allocation satisfies:

\[
\sum_{i=1}^{4} G_i = 120\ \text{seconds}
\]

subject to:

\[
10 \leq G_i \leq 40
\]

where G_i is the green duration assigned to approach i.

Each approach is followed by 3 seconds of yellow and 1 second of all-red clearance. Therefore, the nominal full normal rotation is:

\[
T_{\text{rotation}} = 120 + 4(3) + 4(1) = 136\ \text{seconds}
\]

Emergency priority is requested when the visual emergency confidence and aggregate audio emergency score both exceed 0.65 on the same approach:

\[
C_{\text{visual}} > 0.65
\quad \land \quad
C_{\text{audio, emergency}} > 0.65
\]

The thresholds are software settings. The controller retains the configured yellow and all-red clearance intervals before granting the emergency-priority green phase.

---

# CHAPTER 3 — RESEARCH METHOD

## 3.1 System Requirements and Characteristics

ARGUS is a Python-based software system for simulating adaptive traffic-signal allocation at a four-approach intersection. It processes multiple visual sources, estimates weighted demand, analyses audio from a selected local video, and presents the outputs in a graphical interface.

### 3.1.1 System Requirements

The system requires:

1. Four video inputs representing North, East, South, and West approaches. Inputs may be selected video, webcam, or supported CCTV streams.
2. A trained YOLO26n weight file with the eight configured vehicle classes.
3. A vehicle-count processing stage and a five-frame median filter to reduce short-term fluctuations in displayed detection counts.
4. A class-weight configuration used to calculate approach-level weighted demand.
5. A signal-allocation controller with a 120-second aggregate green budget and 10–40-second per-approach bounds.
6. An audio-classification pipeline using YAMNet embeddings and a Logistic Regression classifier.
7. A dual-modal verification condition for emergency-priority requests.
8. A GUI that displays camera feeds, detections, signal states, countdowns, and audio-confidence information.

### 3.1.2 System Characteristics and Scope

The interface is implemented using CustomTkinter. Video processing uses OpenCV and the Ultralytics detection framework. Audio processing uses pretrained YAMNet embeddings and a Logistic Regression classifier. Separate processing threads are used to keep the interface responsive.

The system is a software simulation. It does not directly control physical traffic lights. Four camera panels represent four simulated approaches; their video sources are not necessarily synchronized views of one physical intersection.

The GUI contains an Auto/Manual selector, but the selector should not be described as two fully implemented controller modes unless the corresponding controller logic is connected and verified in the current source code. In this manuscript, control behaviour is described only where supported by the implemented processing path.

## 3.2 System Architecture and Algorithm Design

### 3.2.1 Input

ARGUS accepts two categories of input:

- **Visual input:** four camera/video sources associated with North, East, South, and West.
- **Audio input:** the soundtrack of a user-selected local video, processed independently in short segments.

**[FIGURE 3.1 PLACEHOLDER — ARGUS SYSTEM ARCHITECTURE]**  
*Insert the original architecture diagram showing visual inputs, YOLO26n detection, vehicle counting, weighted-demand calculation, audio extraction, YAMNet embeddings, Logistic Regression, dual-modal verification, signal controller, and GUI.*

### 3.2.2 Visual Processing

Each visual source is processed by YOLO26n. Detected objects are classified into the eight configured vehicle classes. The system aggregates the detections by approach. A five-frame median filter is applied to reduce fluctuations in the count sequence before the counts are used by the traffic-demand module.

**[FIGURE 3.2 PLACEHOLDER — FOUR-CAMERA GUI VIEW]**  
*Insert a screenshot of the running GUI showing the North, East, South, and West camera panels.*

### 3.2.3 Audio Processing

Audio is taken from the soundtrack of the local video selected by the user. The signal is converted to mono, resampled to 16 kHz, and divided into three-second segments. YAMNet produces 1,024-dimensional embeddings. A Logistic Regression classifier maps the embeddings to ambulance, firetruck, police, or traffic categories. The aggregate emergency score combines the three emergency categories.

**[FIGURE 3.3 PLACEHOLDER — AUDIO CONFIDENCE PANEL]**  
*Insert the GUI screenshot showing audio category scores and the aggregate emergency score.*

### 3.2.4 Weighted-Demand Calculation and Green-Time Allocation

For each approach, the controller calculates:

\[
S_i = \sum_{c=1}^{8} n_{i,c}w_c
\]

The resulting scores are used to allocate the 120-second aggregate green-time budget. The allocation is bounded between 10 and 40 seconds per approach. When a provisional allocation reaches a bound, the remaining share is redistributed among the other eligible approaches. If no vehicles are detected, the controller uses equal allocation.

The score is a software-defined weighted-demand indicator. It should not be labelled as a direct measurement of physical queue length, occupancy, density, or delay.

**[FIGURE 3.4 PLACEHOLDER — GREEN-TIME ALLOCATION FLOWCHART]**  
*Insert the original flowchart for weighted-score calculation, bounded allocation, capped-share redistribution, and zero-demand handling.*

### 3.2.5 Emergency-Priority Logic and Signal Transition

An emergency-priority request requires both visual and audio evidence to exceed the configured threshold on the same approach. The software then processes the priority request through the signal-transition sequence. Yellow and all-red clearance intervals are retained before the next green phase is granted.

The normal signal timing is 3 seconds yellow and 1 second all-red for each approach. The emergency event is handled within the software simulation and does not communicate with physical signal hardware.

The time-compensation component, where applicable, adjusts the ranking of simultaneously confirmed emergency candidates according to elapsed time since an approach last received green. It does not restore missed green time, calculate queue recovery, or establish traffic equilibrium.

**[FIGURE 3.5 PLACEHOLDER — EMERGENCY PREEMPTION STATE DIAGRAM]**  
*Insert the original state diagram showing normal green, yellow, all-red, emergency green, and return to normal operation.*

## 3.3 Software Implementation and Module Integration

The implementation consists of the following functional modules:

| Module | Main function | Main output |
|---|---|---|
| Video input | Reads the four selected camera/video sources | Frames for each approach |
| YOLO26n detector | Detects and classifies eight vehicle categories | Bounding boxes, class labels, confidence |
| Count stabilizer | Applies a five-frame median filter | Stabilized vehicle counts |
| Demand estimator | Applies operational class weights | Weighted score per approach |
| Green allocator | Distributes the 120-second aggregate green budget | Green duration per approach |
| Audio processor | Extracts and segments selected-video audio | Three-second mono segments at 16 kHz |
| YAMNet and classifier | Generates embeddings and classifies audio | Ambulance, firetruck, police, traffic scores |
| Emergency verifier | Checks visual and aggregate audio thresholds on the same approach | Emergency-priority request |
| Signal controller | Applies green, yellow, and all-red transitions | Simulated signal state |
| GUI | Displays camera, detection, signal, and audio information | Operator-facing visualisation |

The visual and audio pipelines operate as separate processing paths. Their outputs are combined by the emergency-verification condition. The GUI presents the processing results and simulated signal states. The system remains a software-based simulation and is not connected to physical traffic-signal controllers.

## 3.4 Experimental Setup and Testing Scenarios

### 3.4.1 Computer Vision Evaluation

The visual model was trained using the eight-class vehicle dataset. The reported training configuration used 640-pixel image size, batch size 16, and a Tesla T4 GPU in the training environment. The GUI inference environment is separate from the training environment and has been reported to use CPU inference on the local computer.

The evaluation reports precision, recall, mAP@0.5, and mAP@0.5:0.95. The dataset audit identified empty-label images; therefore, annotation completeness should be considered when interpreting the validation results. The presence of an empty label does not automatically mean that the image is incorrectly labelled, because some images may legitimately contain no target vehicle.

### 3.4.2 Traffic Allocation Test

The traffic-allocation test records the weighted demand scores for the four approaches and checks whether the resulting green-time allocation satisfies the 120-second total budget and the 10–40-second bounds.

### 3.4.3 Emergency Verification Test

The emergency-verification test checks whether a visual emergency detection and aggregate audio emergency score exceed their configured thresholds on the same approach. A request is treated as confirmed only when both conditions are satisfied.

### 3.4.4 Signal Transition and Return-to-Normal Test

The transition test records the sequence from the emergency request through yellow, all-red, emergency green, and return to normal green. Timing measurements are compared with the configured clearance intervals. A complete traffic-recovery claim requires additional measurements and is not inferred solely from a return to normal signal operation.

## 3.5 Performance Metrics and System Evaluation

The visual model is evaluated using:

- **Precision:** proportion of predicted positive detections that are correct.
- **Recall:** proportion of labelled target objects detected by the model.
- **mAP@0.5:** mean average precision at an IoU threshold of 0.5.
- **mAP@0.5:0.95:** mean average precision averaged over IoU thresholds from 0.5 to 0.95.

The traffic allocator is checked for total-budget compliance and per-approach bounds. Signal-transition evaluation uses the recorded elapsed time between request and state changes. The available experiment does not provide a complete measurement of physical queue length, vehicle delay, throughput, or post-emergency traffic equilibrium.

---

# CHAPTER 4 — RESULTS AND DISCUSSION

## 4.1 Computer Vision Testing

The trained YOLO26n model was evaluated on the eight configured vehicle classes. The reported overall results are shown in Table 4.1.

**Table 4.1. Overall YOLO26n validation metrics**

| Metric | Reported result |
|---|---:|
| Precision | 87.3% |
| Recall | 83.3% |
| mAP@0.5 | 89.5% |
| mAP@0.5:0.95 | 72.1% |

**Table 4.2. Per-class YOLO26n validation metrics**

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

The reported metrics indicate variation between classes. Ambulance and fire-engine detections have comparatively high reported mAP values, while the military, motorcycle, and truck categories show lower values on some metrics. These results should be interpreted in relation to the dataset and annotation quality. They do not establish equivalent performance under every CCTV camera, lighting condition, or road environment.

**[FIGURE 4.1 PLACEHOLDER — TRAINING RESULTS]**  
*Insert the original YOLO26n training-results plot, including precision, recall, mAP, and loss curves if available.*

**[FIGURE 4.2 PLACEHOLDER — NORMALIZED CONFUSION MATRIX]**  
*Insert the original normalized confusion matrix. The reported diagonal entries are 0.94, 0.90, 0.94, 0.96, 0.95, 0.91, 0.93, and 0.83 in the configured class order. These normalized entries should not be presented as recall values.*

## 4.2 Traffic Control Evaluation

A recorded traffic-allocation example at 17:12:42 produced the weighted demand scores and green-time allocation shown in Table 4.3.

**Table 4.3. Example weighted demand and green-time allocation**

| Approach | Weighted demand score | Green allocation |
|---|---:|---:|
| North | 15.64 | 40 s |
| East | 16.98 | 40 s |
| South | 3.00 | 21 s |
| West | 2.66 | 19 s |
| **Total** | **38.28** | **120 s** |

The allocation satisfies the configured total green-time budget of 120 seconds. Each approach also remains within the configured 10–40-second bounds. North and East reach the upper bound, while the remaining green time is distributed to South and West.

The recorded scores were subsequently updated at 17:14:59, as shown in Table 4.4.

**Table 4.4. Updated weighted demand scores**

| Approach | Updated weighted demand score |
|---|---:|
| North | 7.00 |
| East | 17.31 |
| South | 7.65 |
| West | 2.00 |

The difference between the two sets of scores illustrates that the measured input to the allocator changes over time. The example demonstrates the software's weighted-demand calculation and bounded allocation. It does not, by itself, demonstrate reduced queue length, average delay, or increased intersection throughput.

**[FIGURE 4.3 PLACEHOLDER — GUI TRAFFIC ALLOCATION]**  
*Insert a screenshot of the GUI showing the four approach counts, weighted scores, and green-time allocation.*

## 4.3 Emergency Vehicle Priority Testing

The emergency-priority mechanism is designed to require two conditions on the same approach: visual emergency-vehicle confidence greater than 0.65 and aggregate audio emergency score greater than 0.65. The available material describes the configured logic, but a complete set of repeated detection trials, false-positive/false-negative counts, and per-class audio evaluation results is not provided here.

Accordingly, the current evidence supports describing the implemented verification condition, but it is insufficient to calculate an overall emergency-detection accuracy or to claim a measured improvement in emergency-vehicle travel time.

**[FIGURE 4.4 PLACEHOLDER — EMERGENCY PRIORITY GUI STATE]**  
*Insert an original screenshot showing the confirmed emergency approach, visual detection, audio confidence, and signal state.*

## 4.4 System Recovery Evaluation

A software timing observation was recorded on 5 November 2026 between 15:50:06 and 15:52:26 WIB. The event sequence is summarised in Table 4.5.

**Table 4.5. Recorded emergency-priority transition sequence**

| Event | Timestamp |
|---|---|
| Observation begins | 15:50:06 |
| Emergency request on West approach | 15:50:24.009 |
| North enters yellow | After emergency request |
| All-red begins | 15:50:26.050 |
| West emergency green begins | 15:50:27.071 |
| Emergency green ends | 15:50:37.241 |
| Normal North green resumes | 15:50:41.302 |
| Observation ends | 15:52:26 |

The recorded emergency request-to-emergency-green interval was 3.062 seconds. The interval from the end of emergency green to the resumption of normal North green was 4.061 seconds. The total interval from emergency request to normal green resumption was 17.293 seconds.

**Table 4.6. Signal-transition timing summary**

| Transition measurement | Mean | Target | MAE | MAPE | Within ±0.1 s |
|---|---:|---:|---:|---:|---:|
| Yellow to all-red | 2.841 s | 3.000 s | 0.2282 s | 7.61% | 4 of 5 |
| All-red to green | 1.019 s | 1.000 s | 0.0186 s | 1.86% | 5 of 5 |

The all-red-to-green measurements were close to the configured one-second target in the five recorded observations. The yellow-to-all-red mean was below the three-second target, and four of five observations were within ±0.1 seconds. These results describe software timing observations; they should not be interpreted as certification of a physical traffic controller.

Two recorded green allocations met the 120-second total and the 10–40-second per-approach bounds. The first allocation was interrupted by the emergency event, and the second ended shortly after the West green phase. The available records do not contain a complete recovery-duration field; the recovery_elapsed_seconds value is empty. Therefore, the full duration required for traffic recovery and the return to any traffic-equilibrium condition cannot be determined from the current data.

**[FIGURE 4.5 PLACEHOLDER — EMERGENCY TRANSITION TIMELINE]**  
*Insert the original event timeline or signal-state plot showing request, yellow, all-red, emergency green, and normal green resumption.*

## 4.5 Overall System Analysis

The available results demonstrate three limited aspects of ARGUS:

1. The trained YOLO26n model produced the reported validation metrics for the eight configured vehicle classes.
2. The traffic allocator generated a 120-second aggregate green allocation while observing the configured per-approach bounds in the recorded example.
3. The software recorded emergency-priority transitions and clearance intervals, including an observed request-to-emergency-green interval of 3.062 seconds.

The current evidence does not establish reduced real-world traffic delay, shorter emergency travel time, increased throughput, physical queue reduction, or post-emergency traffic equilibrium. Such claims require a controlled comparison with a baseline, repeated trials, complete measurement definitions, and an appropriate physical or validated traffic-simulation environment.

---

# CHAPTER 5 — CONCLUSION

ARGUS was developed as a software-based adaptive traffic-signal simulation for a four-approach intersection. The system combines YOLO26n vehicle detection, study-defined operational vehicle weights, bounded green-time allocation, and dual-modal emergency-vehicle verification using visual detections and audio classification.

The reported YOLO26n validation results were 87.3% precision, 83.3% recall, 89.5% mAP@0.5, and 72.1% mAP@0.5:0.95. In the recorded traffic-allocation example, the controller distributed the 120-second aggregate green budget across the four approaches while satisfying the configured 10–40-second bounds. In the observed emergency event, the request-to-emergency-green interval was 3.062 seconds, and the software retained the yellow and all-red transition sequence.

These findings support the operation of the described software modules under the recorded test conditions. They do not prove improvements in real-world congestion, vehicle delay, throughput, emergency travel time, or physical intersection safety. ARGUS has not been connected to physical traffic-signal controllers, and the available recovery records are insufficient to determine complete traffic-recovery duration.

Further evaluation should include repeated emergency-verification trials, a documented audio test set, a complete dataset-annotation review, controlled baseline comparisons, and a complete recovery measurement protocol. Physical deployment would require separate engineering, safety, and regulatory validation.

---

# REFERENCES

[1] S. Sugiarto, F. Apriandy, Y. Darma, S. M. Saleh, M. Rusdi, and T. Miwa, “Determining Passenger Car Equivalent (PCEs) for Pretimed Signalized Intersections with Severe Motorcycle Composition Using Bayesian Linear Regression,” *PLOS ONE*, vol. 16, no. 9, e0256620, 2021. https://doi.org/10.1371/journal.pone.0256620

[2] T.-P. Hsu, Y.-L. Huang, and M. R. Tsai, “Geometric Layout Dependent Passenger Car Equivalent of Motorcycle,” *Journal of the Eastern Asia Society for Transportation Studies*, vol. 14, pp. 1888–1905, 2022. https://doi.org/10.11175/easts.14.1888

[3] S. Jithender and A. Mehar, “Modeling Passenger Car Unit and Delay at Signalized Intersections in Mixed Traffic Conditions,” *U.Porto Journal of Engineering*, vol. 9, no. 3, pp. 28–40, 2023. https://doi.org/10.24840/2183-6493_009-002_001508

[4] Badan Pusat Statistik Provinsi D.I. Yogyakarta, *Daerah Istimewa Yogyakarta Dalam Angka 2025*. BPS Provinsi D.I. Yogyakarta, 2025.

[5] Á. Serrano, C. Conde, L. J. Rodríguez-Aragón, R. Montes, and E. Cabello, “Computer Vision Application: Real Time Smart Traffic Light,” in *Computer Aided Systems Theory – EUROCAST 2005*, Lecture Notes in Computer Science, vol. 3643, pp. 525–530, 2005. https://doi.org/10.1007/11556985_68

[6] T. Osman, S. S. Psyche, J. M. S. Ferdous, and H. U. Zaman, “Intelligent Traffic Management System for Cross Section of Roads Using Computer Vision,” in *2017 IEEE 7th Annual Computing and Communication Workshop and Conference*, pp. 1–7, 2017. https://doi.org/10.1109/CCWC.2017.7868350

[7] S. K. Abbas et al., “Vision Based Intelligent Traffic Light Management System Using Faster R-CNN,” *CAAI Transactions on Intelligence Technology*, vol. 9, no. 4, pp. 932–947, 2024. https://doi.org/10.1049/cit2.12309

[8] Z. Fahrunnisa, R. Rahmadwati, and R. A. Setyawan, “Adaptive Traffic Light Signal Control Using Fuzzy Logic Based on Real-Time Vehicle Detection from Video Surveillance,” *JITEKI*, vol. 10, no. 2, pp. 235–251, 2024. https://doi.org/10.26555/jiteki.v10i2.28712

[9] J. He, H. Chen, B. Liu, S. Luo, and J. Liu, “Enhancing YOLO for Occluded Vehicle Detection with Grouped Orthogonal Attention and Dense Object Repulsion,” *Scientific Reports*, vol. 14, Article 19650, 2024. https://doi.org/10.1038/s41598-024-70695-x

[10] D. Jayakumar et al., “Emergency Vehicle Classification Using Combined Temporal and Spectral Audio Features with Machine Learning Algorithms,” *Electronics*, vol. 13, no. 19, 3873, 2024. https://doi.org/10.3390/electronics13193873

[11] V.-T. Tran and W.-H. Tsai, “Acoustic-Based Emergency Vehicle Detection Using Convolutional Neural Networks,” *IEEE Access*, vol. 8, pp. 75702–75713, 2020. https://doi.org/10.1109/ACCESS.2020.2988986

[12] V.-T. Tran and W.-H. Tsai, “Audio-Vision Emergency Vehicle Detection,” *IEEE Sensors Journal*, vol. 21, no. 24, pp. 27905–27917, 2021. https://doi.org/10.1109/JSEN.2021.3127893

[13] L. Gabrielli, L. Migliorelli, M. Cantarini, A. Mancini, and S. Squartini, “An Advanced Multimodal Driver-Assistance Prototype for Emergency-Vehicle Detection,” *Integrated Computer-Aided Engineering*, vol. 31, no. 4, pp. 381–399, 2024. https://doi.org/10.3233/ICA-240733

[14] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 779–788, 2016. https://doi.org/10.1109/CVPR.2016.91

[15] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, “Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models,” arXiv:2606.03748, 2026. https://doi.org/10.48550/arXiv.2606.03748

[16] TensorFlow Models, “YAMNet: A Pre-trained Deep Net that Predicts 521 Audio Event Classes,” TensorFlow Models repository. https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

[17] A. I. Majeed, S. Q. Naeem, E. A. Saeed, and A. Al-Shanoon, “Real-Time Adaptive Traffic Signal Control with YOLOv10 and Image Processing,” *Al-Khwarizmi Engineering Journal*, vol. 21, no. 4, pp. 65–81, 2025. https://doi.org/10.22153/kej.2025.09.006

[18] X. Qin and A. M. Khan, “Control Strategies of Traffic Signal Timing Transition for Emergency Vehicle Preemption,” *Transportation Research Part C: Emerging Technologies*, vol. 25, pp. 1–17, 2012. https://doi.org/10.1016/j.trc.2012.04.004

[19] I. Yun, M. Best, and B. Park, “Evaluation of Transition Methods of the 170E and 2070 ATC Traffic Controllers after Emergency Vehicle Preemption,” *Journal of Transportation Engineering*, vol. 134, no. 10, pp. 423–431, 2008. https://doi.org/10.1061/(ASCE)0733-947X(2008)134:10(423)

---

## FIGURE AND IMAGE INSERTION CHECKLIST

| Figure | Required original image | Status in this Markdown |
|---|---|---|
| Figure 3.1 | ARGUS system architecture | Placeholder |
| Figure 3.2 | Four-camera GUI screenshot | Placeholder |
| Figure 3.3 | Audio confidence panel | Placeholder |
| Figure 3.4 | Green-time allocation flowchart | Placeholder |
| Figure 3.5 | Emergency preemption state diagram | Placeholder |
| Figure 4.1 | YOLO26n training results | Placeholder |
| Figure 4.2 | Normalized confusion matrix | Placeholder |
| Figure 4.3 | GUI traffic allocation screenshot | Placeholder |
| Figure 4.4 | Emergency-priority GUI state | Placeholder |
| Figure 4.5 | Emergency transition timeline | Placeholder |

**Image insertion convention:** save the original image in a folder such as images/ beside this Markdown file, then replace the relevant placeholder with a Markdown image reference such as:

    ![Figure 3.1. ARGUS system architecture](images/figure-3-1-architecture.png)

Do not create synthetic screenshots or training plots to replace actual project evidence. Use the original repository or experiment outputs.
