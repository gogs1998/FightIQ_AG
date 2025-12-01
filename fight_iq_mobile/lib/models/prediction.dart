class PredictionResponse {
  final String winner;
  final double confidence;
  final double f1Prob;
  final double f2Prob;
  final bool isValue;
  final String betTarget;
  final double edge;
  final List<String> conformalSet;
  // Master Props
  final String predMethod;
  final String predRound;
  final double trifectaProb;
  final String minOdds;

  PredictionResponse({
    required this.winner,
    required this.confidence,
    required this.f1Prob,
    required this.f2Prob,
    required this.isValue,
    required this.betTarget,
    required this.edge,
    required this.conformalSet,
    required this.predMethod,
    required this.predRound,
    required this.trifectaProb,
    required this.minOdds,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    return PredictionResponse(
      winner: json['winner'],
      confidence: json['confidence'].toDouble(),
      f1Prob: json['f1_prob'].toDouble(),
      f2Prob: json['f2_prob'].toDouble(),
      isValue: json['is_value'],
      betTarget: json['bet_target'],
      edge: json['edge'].toDouble(),
      conformalSet: List<String>.from(json['conformal_set'] ?? []),
      predMethod: json['pred_method'] ?? 'N/A',
      predRound: json['pred_round'] ?? 'N/A',
      trifectaProb: (json['trifecta_prob'] ?? 0.0).toDouble(),
      minOdds: json['min_odds'] ?? 'N/A',
    );
  }
}
