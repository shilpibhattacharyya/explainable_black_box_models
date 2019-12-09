# Explainable Deep Learning - proposal

Data scientists need a way to explain the operations and decisions of black box machine learning algorithms so that more sophisticated and predictive models can be leveraged to increase True Positives and decrease False Positives in a highly regulated space.

## What is the key issue (customer pain) or benefit that motivates the need for this project?
1. Bank operations and decisions, particularly concerning money laundering, are highly regulated. According to federal acts such as the Bank Secrecy Act and Patriot Act banks are required to regulate customer activity to identify when transactions or behavior exhibit suspicious patterns that could be indicative of money laundering or terrorist financing. These instances are reviewed by a set of analysts, who decide whether to waive the instance or escalate it for further review. Any decision made must be thoroughly explained with concrete evidence and reasoning.
2. Machine learning can be highly advantageous to apply in this space. The algorithms are able to identify non-obvious patterns and indicators of money laundering. However, due to the black box nature of most algorithms, regulators do not allow analysts to only cite a machine learning predicted probability as sufficient evidence when waiving or escalating suspicious abnormal behavior. 
3. As stated above, this can lead to a situation in which sophisticated algorithms cannot be leveraged, and True Positives must be prioritized at the expense of generating more False Positives. 
4. We need a way to make the black box models more transparent so that bank analysts can explain their decisions to regulators, and highly predictive models can be safely employed to address the True Positive/False Positive tradeoff. As such, we would be able to leverage more accurate predictions through the use of complex algorithms, while providing understandable evidence as to why that prediction was made. This would allow the development team to:
  a.	Diversify their analytics algorithm stack
  b.	Provide more relevant insights on the UI 
  c.	Leverage the insights and data to create enhanced UI visualizations
  d.	Generate entity specific explanations that would be accepted by financial regulators, enhancing business value and applicability

## How might the results of the project be used after the Challenge?
The machine learning explainability techniques the Jumpstart team identifies and validates would be incorporated into the product to create actionable insights translating how a machine learning algorithm came to a certain decision. These explanations would be printed to the UI so the end user analyst could design a report around the model's reasoning. The data scientists could also use the factors identified in the explainability work as a basis for building new exploratory analytics.

## What are the key technical and business goals?
We hope to identify and implement an innovative approach at making black box models more transparent and explainable to bank regulators and transaction monitoring analysts. The black box models will efficiently generate highly accurate predictions regarding a bank's alerts, cutting down on processing time and the number of alerts that are wrongly passed along for further results.

## What specialized skills might be beneficial for the project?
It would be beneficial to have skills related to machine learning and data science. As a supplement, those with a background in developing visualizations on a UI that help to illustrate algorithmic output would be relevant.

## Any other information you’d like to add (e.g. validation points for the project; customer info available; etc.)?



