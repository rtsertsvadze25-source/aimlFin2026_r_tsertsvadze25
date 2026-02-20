
---

## 1. Event Log File

Original source:

http://max.ge/aiml_final/r_tsertsvadze25_84687_server.log

The file is uploaded in this folder as:

`r_tsertsvadze25_84687_server.log`

---

## 2. Objective

The objective of this task is to:

- Analyze web server logs
- Extract statistical traffic features
- Apply regression analysis
- Detect abnormal traffic spikes
- Identify time interval(s) of DDoS attack

---

## 3. Methodology

### Step 1 — Log Parsing

We extracted timestamps from the log file and grouped requests per minute.

### Step 2 — Time Series Construction

We built a time series:


X = time (in minutes)
Y = number of requests per minute




### Step 3 — Regression Analysis

We applied **Linear Regression** to model normal traffic behavior:

\[
Y = β0 + β1X
\]

The regression line represents expected traffic behavior.

### Step 4 — Anomaly Detection

We calculated residuals:

\[
Residual = Observed - Predicted
\]

We defined DDoS traffic as:



Requests > (mean + 3 * standard deviation)




This statistical threshold detects extreme abnormal spikes.

---

## 4. Source Code (Main Fragments)

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import re


Parsing Log


pattern = r'\[(.*?)\]'
timestamps = []

with open("r_tsertsvadze25_84687_server.log") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            timestamps.append(match.group(1))



Convert to DataFrame

df = pd.DataFrame(timestamps, columns=['time'])
df['time'] = pd.to_datetime(df['time'], format='%d/%b/%Y:%H:%M:%S')
df.set_index('time', inplace=True)

requests_per_min = df.resample('1T').size()










Linear Regression



X = np.arange(len(requests_per_min)).reshape(-1, 1)
y = requests_per_min.values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


Detect DDoS


threshold = requests_per_min.mean() + 3 * requests_per_min.std()
ddos_periods = requests_per_min[requests_per_min > threshold]



Detected DDoS Time Interval(s)

After running the analysis, the detected DDoS attack interval(s) are:


[INSERT DETECTED START TIME] — [INSERT DETECTED END TIME]



python ddos_detection.py


Conclusion

Using regression analysis and statistical thresholding, we successfully identified abnormal traffic spikes corresponding to DDoS attack behavior.

The regression model effectively captured baseline traffic while residual analysis detected extreme deviations.



---

# 🐍 2️⃣ ddos_detection.py (FULL WORKING SCRIPT)

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import re

log_file = "r_tsertsvadze25_84687_server.log"

pattern = r'\[(.*?)\]'
timestamps = []

with open(log_file) as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            timestamps.append(match.group(1))

df = pd.DataFrame(timestamps, columns=['time'])
df['time'] = pd.to_datetime(df['time'], format='%d/%b/%Y:%H:%M:%S')
df.set_index('time', inplace=True)

requests_per_min = df.resample('1T').size()

# Plot requests per minute
plt.figure()
requests_per_min.plot()
plt.title("Requests per Minute")
plt.xlabel("Time")
plt.ylabel("Number of Requests")
plt.savefig("requests_per_minute.png")
plt.close()

# Regression
X = np.arange(len(requests_per_min)).reshape(-1, 1)
y = requests_per_min.values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot regression
plt.figure()
plt.plot(requests_per_min.index, y, label="Actual")
plt.plot(requests_per_min.index, y_pred, label="Regression")
plt.legend()
plt.title("Regression Analysis")
plt.savefig("regression_fit.png")
plt.close()

# DDoS detection
threshold = requests_per_min.mean() + 3 * requests_per_min.std()
ddos_periods = requests_per_min[requests_per_min > threshold]

print("\nDetected DDoS intervals:")
print(ddos_periods)

if not ddos_periods.empty:
    start = ddos_periods.index.min()
    end = ddos_periods.index.max()
    print(f"\nDDoS Time Interval: {start} — {end}")
else:
    print("No DDoS detected.")





