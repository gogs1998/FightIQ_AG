# FightIQ API & Frontend Architecture

## Overview
To upgrade the frontend to **Flutter** (for iOS, Android, and Web), we need to decouple the Python machine learning logic from the UI.

## Architecture
1.  **Backend (Python/FastAPI)**: Serves the model predictions via a REST API.
2.  **Frontend (Flutter)**: Consumes the API and displays the UI.

## 1. Backend (FastAPI)
Create a file `api.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np

app = FastAPI()

# Load Resources
model = joblib.load('ufc_model_elo.pkl')
with open('fighter_db.json', 'r') as f:
    fighter_db = json.load(f)
with open('fighter_elo.json', 'r') as f:
    fighter_elo = json.load(f)
with open('features_elo.json', 'r') as f:
    features = json.load(f)

class FightRequest(BaseModel):
    f1_name: str
    f2_name: str
    f1_odds: float
    f2_odds: float

@app.post("/predict")
def predict(fight: FightRequest):
    # ... (Feature construction logic from app.py) ...
    # Return JSON: {"winner": "Jon Jones", "confidence": 0.88, "is_value": True}
    pass
```

## 2. Frontend (Flutter)
A Flutter app structure would look like this:

```dart
// lib/main.dart
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(FightIQApp());

class FightIQApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FightIQ',
      theme: ThemeData(brightness: Brightness.dark, primaryColor: Colors.red),
      home: PredictionScreen(),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  // Controllers for inputs
  // Function to call API
  Future<void> getPrediction() async {
    final response = await http.post(
      Uri.parse('http://YOUR_API_URL/predict'),
      body: jsonEncode({
        'f1_name': _f1Controller.text,
        'f2_name': _f2Controller.text,
        'f1_odds': double.parse(_odds1Controller.text),
        'f2_odds': double.parse(_odds2Controller.text),
      }),
    );
    // Update state with result
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('FightIQ 🥊')),
      body: Column(
        children: [
          // Inputs for Fighters and Odds
          // Predict Button
          // Result Display Card
        ],
      ),
    );
  }
}
```

## Benefits of Flutter
- **Native Performance**: Smooth animations (60fps).
- **Cross-Platform**: One codebase for iPhone, Android, and Web.
- **Beautiful UI**: Material Design or Cupertino widgets out of the box.
