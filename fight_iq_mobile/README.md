# FightIQ Mobile App

This is the Flutter frontend for the FightIQ prediction engine.

## Prerequisites

1.  **Flutter SDK**: You need to install Flutter to run this app.
    *   Download from: [https://docs.flutter.dev/get-started/install/windows](https://docs.flutter.dev/get-started/install/windows)
    *   Add `flutter/bin` to your PATH environment variable.
    *   Run `flutter doctor` to verify installation.

2.  **Backend API**: The app needs the Python backend to be running.

## Setup

1.  **Install Dependencies**:
    ```bash
    cd fight_iq_mobile
    flutter pub get
    ```

2.  **Run Backend**:
    Open a terminal in the root `FightIQ` folder and run:
    ```bash
    python api.py
    ```
    This will start the server at `http://0.0.0.0:8000`.

3.  **Run App**:
    ```bash
    cd fight_iq_mobile
    flutter run
    ```
    *   **Android Emulator**: Works out of the box (connects to `10.0.2.2`).
    *   **Physical Device**: You may need to update `baseUrl` in `lib/services/api_service.dart` to your PC's local IP address (e.g., `http://192.168.1.X:8000`).

## Project Structure

*   `lib/main.dart`: Main UI and logic.
*   `lib/models/prediction.dart`: Data model for API responses.
*   `lib/services/api_service.dart`: Handles HTTP calls to the backend.
