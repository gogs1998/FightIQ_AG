import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/prediction.dart';

class ApiService {
  // Use 10.0.2.2 for Android emulator, localhost for Web/iOS
  // Note: For physical devices, use your machine's LAN IP
  static String get baseUrl {
    if (const bool.fromEnvironment('dart.library.js_util')) {
      return 'http://127.0.0.1:8003'; // Web
    }
    return 'http://10.0.2.2:8003'; // Android Emulator
  }

  Future<PredictionResponse> predict({
    required String f1Name,
    required String f2Name,
    required double f1Odds,
    required double f2Odds,
  }) async {
    final url = Uri.parse('${ApiService.baseUrl}/predict');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'f1_name': f1Name,
        'f2_name': f2Name,
        'f1_odds': f1Odds,
        'f2_odds': f2Odds,
      }),
    );

    if (response.statusCode == 200) {
      return PredictionResponse.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load prediction: ${response.body}');
    }
  }
}
