import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'dart:async';
import 'dart:math';
import 'dart:convert';
import 'package:http/http.dart' as http;

class DashboardROI extends StatefulWidget {
  const DashboardROI({super.key});

  @override
  State<DashboardROI> createState() => _DashboardROIState();
}

class _DashboardROIState extends State<DashboardROI> {
  // Simulator State
  double totalProduction = 0;
  double defectsDetected = 0;
  double currentROI = 0;
  
  // Constants (from problem_definition.md)
  final double warrantyCostPerUnit = 185.0; // Euros
  final double baselineFNR = 0.025; // 2.5%
  final double modelFNR = 0.001; // 0.1%

  // Real-time stream
  Timer? _timer;
  List<Map<String, dynamic>> recentDetections = [];

  @override
  void initState() {
    super.initState();
    // Simulate real-time production line feed
    _timer = Timer.periodic(const Duration(seconds: 2), (timer) {
      _verifyNewUnit();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _verifyNewUnit() async {
    // 1. Generate Synthetic Sensor Data (simulating the device reading)
    final deviceId = "DW-2024-${10000 + totalProduction.toInt()}";
    
    // Create occasional defects (10% chance for demo purposes)
    bool forceDefect = Random().nextDouble() < 0.10;
    
    final payload = {
      "device_id": deviceId,
      "timestamp": DateTime.now().toIso8601String(),
      "product_line": "EcoLine",
      "vibration_val": forceDefect ? 95.0 + Random().nextDouble() * 10 : 40.0 + Random().nextDouble() * 10,
      "audio_freq_hz": forceDefect ? 1450.0 : 1000.0,
      "temperature": 45.0
    };

    try {
      // 2. Call Backend API
      // Note: use localhost for macOS/Linux. For Android Emulator use 10.0.2.2
      final response = await http.post(
        Uri.parse('http://localhost:8000/verify'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        
        if (mounted) {
          setState(() {
            totalProduction++;
            
            bool isDefective = data['is_defective'];
            
            if (isDefective) {
              defectsDetected++;
              // ROI: We saved money by catching a defect
              // Simple ROI Logic: Each caught defect saves the warranty cost
              currentROI += warrantyCostPerUnit;
            }

            // Update List
            recentDetections.insert(0, {
              ...data,
              "timestamp": DateTime.now(),
            });
            if (recentDetections.length > 10) recentDetections.removeLast();
          });
        }
      } else {
        debugPrint("API Error: ${response.statusCode}");
      }
    } catch (e) {
      debugPrint("Connection Failed: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    final currencyFormat = NumberFormat.simpleCurrency(locale: "en_EU", name: "EUR");

    return Scaffold(
      appBar: AppBar(
        title: const Text('BSH Antigravity - Live ROI'),
        backgroundColor: Colors.blue[900],
        foregroundColor: Colors.white,
      ),
      body: Container(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  "Real-Time Verification Status",
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.green[100],
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.green),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.link, color: Colors.green, size: 16),
                      const SizedBox(width: 4),
                      Text("API Connected", style: TextStyle(color: Colors.green[800], fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // ROI Card (The WOW Factor)
            _buildMetricCard(
              title: "Estimated Warranty Savings",
              value: currencyFormat.format(currentROI),
              icon: Icons.euro,
              color: Colors.green,
              isLarge: true,
            ),
            
            const SizedBox(height: 16),
            
            // Stats Grid
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    title: "Units Verified",
                    value: totalProduction.toInt().toString(),
                    icon: Icons.precision_manufacturing,
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _buildMetricCard(
                    title: "Defects Caught",
                    value: defectsDetected.toInt().toString(),
                    icon: Icons.warning_amber_rounded,
                    color: Colors.orange,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 20),
            
            // XAI / Recent Events Section
            const Text(
              "Recent Detections (Live from Backend)",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: recentDetections.length,
                itemBuilder: (context, index) {
                  final item = recentDetections[index];
                  final bool isDefect = item['is_defective'];
                  final explanations = List<String>.from(item['xai_explanation'] ?? []);
                  final time = DateFormat('HH:mm:ss').format(item['timestamp']);
                  
                  // If defect, show Defect Confidence. If passed, show Safety Confidence (100% - Risk).
                  final double rawConf = item['confidence'];
                  final int displayConf = isDefect 
                      ? (rawConf * 100).toInt() 
                      : ((1.0 - rawConf) * 100).toInt();

                  return Card(
                    child: ListTile(
                      leading: Icon(
                        isDefect ? Icons.error : Icons.check_circle, 
                        color: isDefect ? Colors.red : Colors.green,
                      ),
                      title: Text("${item['device_id']}"),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(isDefect 
                            ? "DEFECT DETECTED ($displayConf% Confidence)" 
                            : "Passed Verification ($displayConf% Safe)"
                          ),
                          if (isDefect && explanations.isNotEmpty)
                            Text("Explanations: ${explanations.join(", ")}", style: const TextStyle(fontSize: 12, color: Colors.grey)),
                        ],
                      ),
                      trailing: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(time),
                          if (isDefect) ...[
                            const SizedBox(width: 8),
                            ElevatedButton.icon(
                              onPressed: () async {
                                // Real Backend Reporting
                                try {
                                  final reportPayload = {
                                    "device_id": item['device_id'],
                                    "timestamp": DateTime.now().toIso8601String(),
                                    "reason": explanations.isNotEmpty ? explanations.first : "Manual Flag",
                                    "confidence": item['confidence'],
                                    "user_action": "manual_qa_flag"
                                  };

                                  final resp = await http.post(
                                    Uri.parse('http://localhost:8000/report'),
                                    headers: {"Content-Type": "application/json"},
                                    body: jsonEncode(reportPayload),
                                  );
                                  
                                  if (context.mounted) {
                                    if (resp.statusCode == 200) {
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(
                                          content: Text("✅ Report Logged: ${item['device_id']}"),
                                          backgroundColor: Colors.green,
                                        )
                                      );
                                    } else {
                                      throw Exception("API Error ${resp.statusCode}");
                                    }
                                  }
                                } catch (e) {
                                   if (context.mounted) {
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text("❌ Failed: $e"), backgroundColor: Colors.red)
                                    );
                                   }
                                }
                              },
                              icon: const Icon(Icons.send_and_archive, size: 16),
                              label: const Text("Report"),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.red[50],
                                foregroundColor: Colors.red,
                                elevation: 0,
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricCard({
    required String title,
    required String value,
    required IconData icon,
    required Color color,
    bool isLarge = false,
  }) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [color.withOpacity(0.1), color.withOpacity(0.05)],
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: isLarge ? 40 : 30),
            const SizedBox(height: 8),
            Text(
              title,
              style: TextStyle(
                color: Colors.grey[700],
                fontSize: isLarge ? 16 : 14,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: TextStyle(
                color: color,
                fontSize: isLarge ? 32 : 24,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
