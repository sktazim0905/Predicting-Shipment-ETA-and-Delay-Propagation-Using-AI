# **Aiqon AI Internship – Take-Home Assignment (3 Days) Predicting Shipment ETA and Delay Propagation Using AI** 

## **Overview** 

In global logistics, estimated time of arrival (ETA) is one of the most critical signals for planning and coordination. However, ETA is rarely static. Delays at one stage of a shipment, such as late vessel arrival, port congestion, or customs clearance, often propagate downstream and affect the entire delivery timeline. 

These cascading delays are difficult to model because they depend on multiple interacting factors, including transport schedules, port operations, and external disruptions. 

At Aiqon, we aim to build systems that not only predict ETA, but also understand how delays propagate across the shipment lifecycle. This assignment evaluates your ability to model such dynamics in a structured and interpretable way. 

## **Problem Statement** 

You are given a dataset representing shipment journeys. Each shipment may include multiple stages such as origin departure, port arrival, customs clearance, and final delivery. 

Your task is to design and implement a system that can estimate the expected arrival time of a shipment and identify the likelihood of delays. 

Your system should: 

1. Predict the final ETA of a shipment, 

2. Estimate the likelihood or magnitude of delay, 

3. Explain which factors contribute most to the predicted delay. 

The focus is not only on prediction accuracy, but also on how well your system captures the relationship between different stages of the shipment. 

## **Dataset** 

You may use publicly available datasets or construct a synthetic dataset. 

If using synthetic data, you should simulate shipment journeys with fields such as: 

- origin and destination, 

- scheduled departure and arrival times, 

- actual timestamps for intermediate stages, 



- port processing times, 

- delays at each stage, 

- external factors such as congestion or weather indicators. 

A dataset of approximately 100–300 shipment records is sufficient. 

If you choose to use real data, you may explore maritime or transport datasets available on platforms such as Kaggle or AIS-based sources. 

## **What You Are Expected to Build** 

You should begin by defining how ETA is calculated in your system and how delays are measured. This may involve computing differences between scheduled and actual times or modeling time increments across stages. 

You should identify the key features that influence delays. These may include stage-level delays, port-related factors, or temporal patterns. Strong solutions often construct features that capture interactions between stages rather than treating each stage independently. 

You should then implement a model or method that predicts final ETA or delay. This could involve regression, time-series modeling, or a structured rule-based approach. 

A key requirement is that your system should provide explanations. For example, your output may indicate that delays at a specific port or stage contributed significantly to the final delay. 

## **Suggested Challenges to Address** 

You may consider challenges such as: 

- propagation of delays across multiple stages, 

- incomplete or missing intermediate data, 

- variability across different routes, 

- interaction between congestion and processing times. 

You are not required to address all of these, but demonstrating awareness of such issues is valuable. 

## **Evaluation** 

You should define how your system is evaluated. For example, you may use error metrics such as mean absolute error for ETA prediction or classification metrics if you frame delay prediction as a binary problem. 

If using synthetic data, clearly describe how delays were introduced and how ground truth was defined. 



## **Deliverables** 

Your submission should include: 

- a code repository with clear instructions, 

- a technical report (2–3 pages) describing your approach, assumptions, and results, 

- an optional demo (maximum 5 minutes). 

## **What We Will Evaluate** 

We will evaluate your submission based on your understanding of the problem, the quality of your feature design, the appropriateness of your approach, and your ability to explain your predictions. 

We are particularly interested in whether your system reflects real-world logistics behavior rather than purely abstract modeling. 

## **Constraints** 

You should complete this assignment within three days. The solution should be focused and clearly justified. Any assumptions made should be explicitly stated. 

## **Submission Instructions** 

Please submit: 

- GitHub repository link, 

- PDF report, 

- optional demo. 

Submission  email:  info@aiqon.in 

## **Closing Note** 

ETA prediction is not just a forecasting problem. It is a system-level challenge that requires understanding how delays emerge and propagate across multiple stages. 

At Aiqon, we are interested in solutions that move beyond isolated predictions and provide actionable insights into the causes of delays. 



_Aiqon_ 

