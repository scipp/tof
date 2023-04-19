# tof

A simple tool to create time-of-flight chopper cascade diagrams

## Installation

```sh
pip install tof
```

## Example

```Python
import numpy as np
import tof

pulse = tof.Pulse(kind='ess', neutrons=1_000_000)

pulse.plot()
```

![pulse](docs/_static/pulse.png)

```Python
choppers = [
    tof.Chopper(
        frequency=70,
        open=np.array([83.71, 140.49, 193.26, 242.32, 287.91, 330.3]) + 15.0,
        close=np.array([94.7, 155.79, 212.56, 265.33, 314.37, 360.0]) + 15.0,
        phase=47.10,
        distance=6.6,
        unit="deg",
        name="WFM1",
    ),
    tof.Chopper(
        frequency=70,
        open=np.array([65.04, 126.1, 182.88, 235.67, 284.73, 330.00]) + 15.0,
        close=np.array([76.03, 141.4, 202.18, 254.97, 307.74, 360.0]) + 15.0,
        phase=76.76,
        distance=7.1,
        unit="deg",
        name="WFM2",
    ),
    tof.Chopper(
        frequency=56,
        open=np.array([74.6, 139.6, 194.3, 245.3, 294.8, 347.2]),
        close=np.array([95.2, 162.8, 216.1, 263.1, 310.5, 371.6]),
        phase=62.40,
        distance=8.8,
        unit="deg",
        name="Frame-overlap 1",
    ),
    tof.Chopper(
        frequency=28,
        open=np.array([98.0, 154.0, 206.8, 254.0, 299.0, 344.65]),
        close=np.array([134.6, 190.06, 237.01, 280.88, 323.56, 373.76]),
        phase=12.27,
        distance=15.9,
        unit="deg",
        name="Frame-overlap 2",
    ),
]

detectors = [
    tof.Detector(distance=23.0, name='monitor'),
    tof.Detector(distance=32.0, name='detector'),
]

model = tof.Model(choppers=choppers, pulse=pulse, detectors=detectors)

model.run()

model.plot(max_rays=10000)
```

![model](docs/_static/model.png)

```Python
detectors[1].plot()
```

![detector](docs/_static/detector.png)
