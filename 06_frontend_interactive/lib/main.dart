import 'package:flutter/material.dart';
import 'screens/dashboard_roi.dart';

void main() {
  runApp(const BSHVerificationApp());
}

class BSHVerificationApp extends StatelessWidget {
  const BSHVerificationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BSH Verification Dashboard',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1976D2),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1976D2),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const DashboardROI(),
      debugShowCheckedModeBanner: false,
    );
  }
}
