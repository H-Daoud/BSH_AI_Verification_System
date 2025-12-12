import 'package:flutter/material.dart';

class VerifyDetailScreen extends StatelessWidget {
  final String deviceId;
  
  const VerifyDetailScreen({super.key, required this.deviceId});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Verification: \$deviceId'),
      ),
      body: Center(
        child: Text('Verification details for \$deviceId'),
      ),
    );
  }
}
