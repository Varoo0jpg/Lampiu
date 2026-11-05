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

**[Insert Figure 3.X — ARGUS System Architecture Diagram here.]**

The software architecture of the ARGUS system integrates its core processing modules within a Python-based environment. Vehicle detection is performed using a trained YOLO26n model through the Ultralytics framework and OpenCV, targeting eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck, with a confidence threshold of 0.15 and an NMS IoU threshold of 0.60. A five-frame median filter is applied to stabilize detection counts against short-term fluctuations. Meanwhile, audio classification uses pretrained YAMNet embeddings combined with a Logistic Regression classifier. It processes three-second mono audio segments extracted from the soundtrack of a user-selected local video at a 16 kHz sampling rate. The resulting emergency scores are used for dual-modal verification.

Traffic-load estimation maps detected vehicle counts to operational class weights to calculate a cumulative weighted-demand score for each approach. These scores determine green-time allocation within a 120-second aggregate green-time budget, with each approach receiving 10–40 seconds or an equal share under zero demand. Signal transitions include a three-second yellow interval and a one-second all-red clearance phase. Emergency priority is requested when visual detection confidence and the aggregate audio emergency score both exceed 0.65 on the same approach. The CustomTkinter GUI displays video feeds, detection statistics, signal phases, countdown timers, and audio-confidence metrics, while supporting CSV snapshot exports for quantitative evaluation. Separate execution threads handle camera processing and audio analysis to maintain interface responsiveness. ARGUS currently operates as a software-based simulation and is not connected to physical traffic-signal controllers.

## 3.3. Software Implementation and Module Integration

### 3.3.1. Software Implementation

#### 3.3.1.1. YOLO26n-Based Vehicle Detection

Vehicle detection is implemented using a trained YOLO26n model integrated through the Ultralytics library and OpenCV. The model identifies eight vehicle classes: ambulance, bus, car, fire engine, military vehicle, motorcycle, police car, and truck. Video frames from four approaches (North, East, South, and West) are processed using a confidence threshold of 0.15 and an IoU threshold of 0.60. Detected vehicles are classified and counted according to their categories. A five-frame median filter is applied to reduce fluctuations in detection counts before the results are transferred to the traffic-load estimation module.

#### 3.3.1.2. YAMNet-Based Audio Classification and Siren Detection

Audio classification is implemented using pretrained YAMNet embeddings combined with a Logistic Regression classifier. Audio is extracted from a user-selected local video and processed in three-second segments at a sampling rate of 16 kHz. YAMNet generates 1,024-dimensional audio embeddings, which are classified into four categories: ambulance, firetruck, police, and traffic. The emergency score is calculated from the combined classification scores of the three emergency-vehicle categories. The resulting score is displayed through the Audio Confidence panel and used for emergency-vehicle verification.

#### 3.3.1.3. PCU-Based Traffic Load Estimation and Signal Control

Traffic load is estimated by multiplying the detected vehicle counts by their respective operational weighting values. The weighted values are then summed to obtain the traffic-load score for each approach. These scores determine the allocation of green-light duration within a 120-second aggregate green-time budget. The configured allocation has a minimum duration of 10 seconds and a maximum duration of 40 seconds for each approach. When no vehicles are detected, the green-time budget is divided equally among the four approaches.

Signal transitions consist of green, yellow (3 seconds), and all-red (1 second) phases. Since the 120 seconds refers to the total green allocation, the nominal full normal rotation, including four yellow and four all-red intervals, is 136 seconds. Emergency prioritization is activated when visual and audio confidence scores on the same approach both exceed their respective thresholds of 0.65.

#### 3.3.1.4. Graphical User Interface and Research Data Logging

The graphical user interface is developed using CustomTkinter to display camera feeds, vehicle counts, traffic-signal states, countdown timers, and audio-confidence scores. The interface includes an Auto/Manual selector and controls for the simulation. **The presence of the selector in the GUI should not be interpreted as proof of two independently implemented controller modes: the current controller logic does not consume the mode variable.** The GUI also displays vehicle-detection sensitivity settings and approach information where available.

The research data-logging module records traffic conditions, including vehicle classifications, weighted traffic scores, signal phases, and active approaches. Recorded data can be exported for further analysis. Detailed CSV contents are outside the scope of this manuscript audit except where they support the GUI's operational functions.

## 3.4. Experimental Setup and Testing Scenarios

### 3.4.1. Simulation Environment & Configuration

The test environment represents four approaches: North, East, South, and West. For each test, record the video source assigned to each approach, computer specifications, operating system, Python and package versions, YOLO26n checkpoint, and GUI settings. For audio tests, also record the selected local video and its assigned approach.

### 3.4.2. Heterogeneous Traffic Density & PCU Weighting Protocol

Prepare low-, medium-, and high-demand scenarios with known vehicle counts and different class compositions. Calculate the expected approach scores manually using Table 3.1, then compare them with the scores displayed by ARGUS. Record the four green durations and verify that each is between 10 and 40 seconds and that their sum is 120 seconds. Include an all-zero-demand scenario to verify equal allocation.

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

Compare ARGUS weighted scores with manually calculated scores from known vehicle counts and Table 3.1. For signal allocation, report each approach's green duration and verify the total allocation and per-approach limits:

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

Hasil deteksi kendaraan dan metrik akurasi YOLO26n.

**Reported overall validation metrics (available project results):**

| Metric | Result |
|---|---:|
| Precision | 87.3% |
| Recall | 83.3% |
| mAP@0.5 | 89.5% |
| mAP@0.5:0.95 | 72.1% |

**Reported per-class metrics:**

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

These are the reported validation metrics. They should be interpreted with the dataset-annotation audit in mind, which identified empty-label images requiring visual review. An empty label is not automatically an annotation error, but possible missing annotations can affect the interpretation of validation results.

**[Insert the original YOLO26n training-results plot and normalized confusion matrix here, if those figures are included in the source document.]**

## 4.2. Traffic Control Evaluation

The traffic control evaluation was conducted to examine how ARGUS distributes green-light duration among the four approaches based on their weighted traffic scores. The system allocates a total of 120 seconds of green time, with a minimum duration of 10 seconds and a maximum duration of 40 seconds for each approach.

Based on the simulation results recorded in banyu4.csv, the green-time allocation for one normal cycle is presented in Table 4.1. **The 120 seconds refers to the aggregate green-time allocation, not the complete signal rotation. Including four 3-second yellow intervals and four 1-second all-red intervals, the nominal full normal rotation is 136 seconds.**

**Table 4.1. Green-Time Allocation for Each Approach**

[Retain the original Table 4.1 from the DOCX here. Its table cells were not available in the extracted text.]

The results show that the total green-time allocation reaches 120 seconds. Each approach receives a duration within the configured minimum and maximum limits. The East approach receives the longest green duration of 38 seconds, while the South approach receives the shortest duration of 22 seconds.

The simulation also records the signal transition sequence from green to yellow, followed by all-red before the next approach receives a green signal. The configured yellow and all-red intervals are 3 seconds and 1 second, respectively.

These results demonstrate that ARGUS can distribute green-light duration within the configured constraints and execute the normal signal transition sequence in the software simulation. However, these findings evaluate the timing allocation mechanism and do not directly demonstrate a reduction in vehicle queues, waiting time, or traffic congestion.

## 4.3. Emergency Vehicle Priority Testing

Deteksi visual, audio, konfirmasi ganda, dan latency.

[The original manuscript contains this section heading and topic note but does not provide the corresponding test results. Add only results supported by recorded visual/audio test cases.]

## 4.4. System Recovery Evaluation

Kondisi lalu lintas setelah prioritas darurat.

[The original manuscript contains this section heading and topic note but does not provide a complete recovery measurement. The available project notes do not establish a complete recovery duration or traffic-equilibrium result.]

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

# AUDIT CHANGE LOG — COMPARISON WITH THE ORIGINAL DOCX

The original chapter order, section order, research questions, objectives, main terminology, and reference-list order have been retained. The following are the substantive changes made in this Markdown audit copy.

| Original location | Change made | Reason |
|---|---|---|
| 3.2 and 3.3.2 | Consolidated the architecture explanation into one Section 3.2 with an architecture-diagram placeholder and two paragraphs; removed the repetitive Module Integration subsection and retained its distinct thread-processing detail in Section 3.2 | Reduces repetition while preserving the processing pipelines, dual-modal verification, GUI, CSV logging, and software-simulation limitation |\n| Abstract | Removed the statement that the manuscript has no completed experimental results; retained the original system description and clarified that it is a software simulation | The project has reported model-validation and software-timing results, but these do not establish field effectiveness |
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
2. Restore the original Table 4.1 cell values from the DOCX if they are embedded as an image or floating object; the extracted document text did not contain the table cells.
3. Add the actual emergency-priority test results before completing Section 4.3.
4. Add a defined and fully measured recovery result before completing Section 4.4.
5. Complete Chapter 5 only after the final Chapter 4 results are settled.

**Scope note:** CSV contents were not independently audited or used to replace the original Chapter 4.2 result. The original figure/table placement should be retained when transferring this Markdown back into the formatted paper.
