# loan_prediction_Logistic_Regression
The model is a Logistic Regression classifier.Its job is to answer one question:“Will this customer fully pay the loan or default?”It does this by:Turning customer details into numbers,Calculating a risk score  Converting that score into a probability,Making a yes/no decision.

Step 1:What goes into the model?

For each customer, the model receives values like:
Interest rate
Income
FICO score
Debt-to-income ratio
Revolving balance
Past delinquencies
Purpose of loan

After scaling, all values are on a comparable scale.

Step 2: Risk score calculation (MOST IMPORTANT)

Inside Logistic Regression, this happens:

Risk Score=b0​+b1​x1​+b2​x2​+⋯+bn​xn

​
Example intuition:

High interest rate → increases risk score
Low FICO → increases risk score
High income → decreases risk score
This is the “decision logic”.

Step 3: Convert score to probability (Sigmoid)
The risk score is passed through a sigmoid function:P(default)=1/1+e−score​
This:
Converts any number into 0–1
Interprets it as probability of default

Example:
Score = 2.0 → 88% default
Score = -2.0 → 12% default

Step 4: Decision threshold

By default:
if probability >= 0.5:
    Default (1)
else:
    Fully paid (0)

Step 5: Why it chooses “Eligible” or “Not Eligible”
Let’s say your model prints:
Fully Paid: 72%
Default: 28%
Since default probability < 50%
Loan Approved (Eligible)

If:
Fully Paid: 35%
Default: 65%
Loan Rejected (High Risk)

sk)
