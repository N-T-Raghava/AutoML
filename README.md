# SmartML

**SmartML** is a lightweight AutoML library to automatically choose the best algorithm for your dataset.

## Features

- Supports classification and regression
- Auto model selection
- Performance graphs

## Usage

```python
from smartml import SmartML
model = SmartML(df, target='target_column')
model.run()
```

## Install

```bash
pip install -r requirements.txt
```
