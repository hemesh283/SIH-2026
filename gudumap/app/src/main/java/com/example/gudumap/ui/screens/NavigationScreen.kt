package com.example.gudumap.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.gudumap.ui.components.MapView
import com.example.gudumap.viewmodel.NavigationViewModel
import java.util.Locale

@Composable
fun NavigationScreen(
    navViewModel: NavigationViewModel = viewModel(factory = NavigationViewModel.Factory)
) {
    val context = LocalContext.current
    val navState by navViewModel.state.collectAsState()

    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        permissionGranted = granted
        if (granted) {
            // Fix (PROJECT_STATUS.md §13/14): previously this only updated the local UI flag
            // above -- location updates were never actually (re-)registered after a late grant.
            navViewModel.retryLocationUpdatesIfNeeded()
        }
    }

    // Also catch permission granted via system Settings while backgrounded (e.g. a judge saying
    // "wait, grant it again" mid-demo) -- re-check on every resume, not just the in-app dialog.
    val lifecycleOwner = LocalLifecycleOwner.current
    val latestViewModel by rememberUpdatedState(navViewModel)
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                permissionGranted = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED
                latestViewModel.retryLocationUpdatesIfNeeded()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    var blackoutControlStage by remember { mutableStateOf(0) } // 0: GNSS AVAILABLE, 1: START GNSS BLACKOUT

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8F9FA))
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {

        // ========================================================
        // HEADER
        // ========================================================
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "GUDUMAP",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = Color(0xFF1E293B)
                )
                Text(
                    text = "Intelligent Dead Reckoning Navigation",
                    fontSize = 13.sp,
                    color = Color(0xFF64748B)
                )
            }

            // Mode Badge
            val badgeMode = if (navState.blackoutMode) "DEAD RECKONING" else "GNSS"
            Box(
                modifier = Modifier
                    .background(
                        if (navState.blackoutMode) Color(0xFFFEF2F2) else Color(0xFFF0FDF4),
                        shape = RoundedCornerShape(8.dp)
                    )
                    .padding(horizontal = 10.dp, vertical = 5.dp)
            ) {
                Text(
                    text = badgeMode,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (navState.blackoutMode) Color(0xFFDC2626) else Color(0xFF16A34A)
                )
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // ========================================================
        // GNSS RECOVERY BANNER (PART 10 & 14)
        // ========================================================
        if (navState.gnssRecovered) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFECFDF5)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "GNSS RECOVERED",
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                            color = Color(0xFF047857)
                        )
                        Text(
                            text = String.format(Locale.US, "Duration: %.1f s", navState.blackoutMetrics.blackoutDurationSeconds),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFF065F46)
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = String.format(Locale.US, "DR: %.1f m", navState.blackoutMetrics.drDistance),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFF065F46)
                        )
                        Text(
                            text = String.format(Locale.US, "GNSS Ref: %.1f m", navState.blackoutMetrics.gnssReferenceDistance),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFF065F46)
                        )
                        Text(
                            text = String.format(Locale.US, "Error: %.2f m", navState.recoveryDriftMeters),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFF065F46)
                        )
                        Text(
                            text = String.format(Locale.US, "Drift: %.1f %%", navState.recoveryErrorPercent),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFF065F46)
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        // ========================================================
        // BLACKOUT STATUS CARD (PART 6)
        // ========================================================
        if (navState.blackoutMode) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFEF2F2)),
                shape = RoundedCornerShape(14.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "GNSS BLACKOUT",
                            fontSize = 15.sp,
                            fontWeight = FontWeight.ExtraBold,
                            color = Color(0xFFDC2626)
                        )
                        val durSec = navState.blackoutDurationSeconds.toInt()
                        val mm = durSec / 60
                        val ss = durSec % 60
                        Text(
                            text = String.format(Locale.US, "Duration: %02d:%02d", mm, ss),
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF991B1B)
                        )
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricTile(
                            title = "DR Distance",
                            value = String.format(Locale.US, "%.1f m", navState.blackoutMetrics.drDistance),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "Max Error",
                            value = String.format(Locale.US, "%.1f m", navState.blackoutMetrics.maximumPositionErrorMeters),
                            modifier = Modifier.weight(1f)
                        )
                        MetricTile(
                            title = "ML Latency",
                            value = "${navState.mlInferenceLatencyMs} ms",
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricTile(
                            title = "Confidence Radius",
                            value = String.format(Locale.US, "±%.1f m", navState.uncertaintyRadiusMeters),
                            modifier = Modifier.weight(1f)
                        )
                        HeadingConfidenceTile(
                            confidence = navState.headingConfidence,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    if (navState.gnssGroundTruthLat != null && navState.gnssGroundTruthLon != null) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Color(0xFFFEE2E2), shape = RoundedCornerShape(8.dp))
                                .padding(10.dp)
                        ) {
                            Column {
                                Text(
                                    text = "GNSS Ground Truth (Evaluation only - NOT used for navigation)",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFFB91C1C)
                                )
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(
                                    text = String.format(Locale.US, "Lat: %.6f, Lon: %.6f", navState.gnssGroundTruthLat, navState.gnssGroundTruthLon),
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Medium,
                                    color = Color(0xFF7F1D1D)
                                )
                            }
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        // ========================================================
        // NAVIGATION STATUS (PART 3 & PART 5)
        // ========================================================
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(14.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "NAVIGATION STATUS",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF64748B),
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(10.dp))

                val gnssDisplay = when {
                    navState.blackoutMode -> "BLACKOUT"
                    navState.gnssNavigationMode == "GNSS_RECOVERY" -> "RECOVERING"
                    else -> navState.gnssStatus
                }
                val navDisplay = if (navState.blackoutMode) "DR" else "GNSS"

                StatusRow(label = "GNSS", value = gnssDisplay, isGood = gnssDisplay == "AVAILABLE")
                StatusRow(label = "Navigation", value = navDisplay, isGood = navDisplay == "GNSS")
                StatusRow(label = "Motion", value = navState.motionState, isGood = true)
                val gateDisplay = "${navState.latestGateAction} (A:${navState.acceptedCount} C:${navState.clampedCount} R:${navState.rejectedCount})"
                StatusRow(label = "ML Gate", value = gateDisplay, isGood = navState.latestGateAction == "ACCEPTED")
                StatusRow(label = "ML", value = navState.mlStatus, isGood = navState.mlStatus == "ACTIVE")
                StatusRow(label = "EKF", value = navState.ekfStatus, isGood = navState.ekfStatus == "ACTIVE")
                StatusRow(label = "MAP", value = navState.mapStatus, isGood = true)
                StatusRow(label = "OFFLINE MAP", value = navState.offlineMapStatus, isGood = navState.offlineMapStatus == "AVAILABLE")

                if (navState.currentRoadName.isNotBlank()) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Road: ${navState.currentRoadName}",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color(0xFF2563EB)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Prominent Blackout Control Button (Requirement 5)
                when {
                    !navState.hasGpsFix -> {
                        // Never allow blackout entry without a real GPS fix -- surface this
                        // plainly instead of silently falling back to any default position
                        // (Fix 2, PROJECT_STATUS.md §11).
                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF64748B)),
                            shape = RoundedCornerShape(10.dp),
                            enabled = false,
                            onClick = {}
                        ) {
                            Text(
                                text = "WAITING FOR GPS FIX...",
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    navState.blackoutMode -> {
                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626)),
                            shape = RoundedCornerShape(10.dp),
                            onClick = {
                                navViewModel.setBlackoutMode(false)
                                blackoutControlStage = 0
                            }
                        ) {
                            Text(
                                text = "GNSS BLACKOUT ACTIVE (TAP TO END)",
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    navState.gnssNavigationMode == "GNSS_RECOVERY" -> {
                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706)),
                            shape = RoundedCornerShape(10.dp),
                            onClick = {}
                        ) {
                            Text(
                                text = "RECOVERING GNSS...",
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    blackoutControlStage == 1 -> {
                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626)),
                            shape = RoundedCornerShape(10.dp),
                            onClick = {
                                navViewModel.setBlackoutMode(true)
                                blackoutControlStage = 0
                            }
                        ) {
                            Text(
                                text = "START GNSS BLACKOUT",
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    else -> {
                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A34A)),
                            shape = RoundedCornerShape(10.dp),
                            onClick = {
                                blackoutControlStage = 1
                            }
                        ) {
                            Text(
                                text = "GNSS AVAILABLE",
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ========================================================
        // MAP (PART 3 & PART 11)
        // ========================================================
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(14.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(10.dp)) {
                MapView(
                    latitude = navState.latitude,
                    longitude = navState.longitude,
                    headingDeg = navState.headingDeg,
                    mapStatus = navState.mapStatus,
                    offlineMapStatus = navState.offlineMapStatus,
                    roadName = navState.currentRoadName,
                    blackoutMode = navState.blackoutMode,
                    naiveLatitude = navState.naiveLatitude,
                    naiveLongitude = navState.naiveLongitude,
                    uncertaintyRadiusMeters = navState.uncertaintyRadiusMeters
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ========================================================
        // POSITION (PART 3)
        // ========================================================
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(14.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "POSITION",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF64748B),
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text(text = "Latitude", fontSize = 11.sp, color = Color(0xFF94A3B8))
                        Text(
                            text = String.format(Locale.US, "%.6f", navState.latitude),
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1E293B)
                        )
                    }
                    Column {
                        Text(text = "Longitude", fontSize = 11.sp, color = Color(0xFF94A3B8))
                        Text(
                            text = String.format(Locale.US, "%.6f", navState.longitude),
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1E293B)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ========================================================
        // NAVIGATION METRICS (PART 3)
        // ========================================================
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(14.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "NAVIGATION METRICS",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF64748B),
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MetricTile(
                        title = "Speed",
                        value = String.format(Locale.US, "%.1f km/h", navState.speedKmh),
                        modifier = Modifier.weight(1f)
                    )
                    MetricTile(
                        title = "Heading",
                        value = String.format(Locale.US, "%.0f°", navState.headingDeg),
                        modifier = Modifier.weight(1f)
                    )
                    MetricTile(
                        title = "Distance",
                        value = String.format(Locale.US, "%.1f m", navState.distanceMeters),
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MetricTile(
                        title = "DR Error",
                        value = String.format(Locale.US, "%.2f m", navState.positionErrorMeters),
                        modifier = Modifier.weight(1f)
                    )
                    MetricTile(
                        title = "Drift",
                        value = String.format(Locale.US, "%.2f %%", navState.driftPercentage),
                        modifier = Modifier.weight(1f)
                    )
                    MetricTile(
                        title = "ML Inference",
                        value = "${navState.mlInferenceLatencyMs} ms",
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ========================================================
        // SENSOR STATUS (PART 3)
        // ========================================================
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(14.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "SENSOR STATUS",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF64748B),
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(10.dp))

                SensorStatusRow(name = "Accelerometer", isActive = navState.accelerometerActive)
                SensorStatusRow(name = "Gyroscope", isActive = navState.gyroscopeActive)
                SensorStatusRow(name = "Magnetometer", isActive = navState.magnetometerActive)
            }
        }

        // Location permission request if not granted
        if (!permissionGranted) {
            Spacer(modifier = Modifier.height(14.dp))
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2563EB)),
                shape = RoundedCornerShape(10.dp),
                onClick = {
                    permissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
                }
            ) {
                Text("GRANT LOCATION PERMISSION", fontWeight = FontWeight.Bold)
            }
        }

        Spacer(modifier = Modifier.height(30.dp))
    }
}

@Composable
private fun StatusRow(label: String, value: String, isGood: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, fontSize = 13.sp, color = Color(0xFF475569))
        Box(
            modifier = Modifier
                .background(
                    if (isGood) Color(0xFFECFDF5) else Color(0xFFFEF2F2),
                    shape = RoundedCornerShape(6.dp)
                )
                .padding(horizontal = 8.dp, vertical = 2.dp)
        ) {
            Text(
                text = value,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = if (isGood) Color(0xFF047857) else Color(0xFFDC2626)
            )
        }
    }
}

@Composable
private fun SensorStatusRow(name: String, isActive: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = name, fontSize = 13.sp, color = Color(0xFF475569))
        Box(
            modifier = Modifier
                .background(
                    if (isActive) Color(0xFFECFDF5) else Color(0xFFFEF2F2),
                    shape = RoundedCornerShape(6.dp)
                )
                .padding(horizontal = 8.dp, vertical = 2.dp)
        ) {
            Text(
                text = if (isActive) "ACTIVE" else "INACTIVE",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = if (isActive) Color(0xFF047857) else Color(0xFFDC2626)
            )
        }
    }
}

@Composable
private fun MetricTile(title: String, value: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Color(0xFFF1F5F9)),
        shape = RoundedCornerShape(10.dp)
    ) {
        Column(
            modifier = Modifier.padding(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = title, fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF64748B))
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = value, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color(0xFF0F172A))
        }
    }
}

/**
 * Surfaces Android's own magnetometer/rotation-vector reliability signal for heading, so a
 * degraded compass (common inside a vehicle chassis or a tunnel/parking-garage's rebar --
 * exactly this project's blackout scenario) is visible rather than silently assumed fine.
 */
@Composable
private fun HeadingConfidenceTile(confidence: String, modifier: Modifier = Modifier) {
    val color = when (confidence) {
        "HIGH" -> Color(0xFF16A34A)
        "MEDIUM" -> Color(0xFFD97706)
        else -> Color(0xFFDC2626) // LOW or UNRELIABLE
    }
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Color(0xFFF1F5F9)),
        shape = RoundedCornerShape(10.dp)
    ) {
        Column(
            modifier = Modifier.padding(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = "Heading Conf.", fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF64748B))
            Spacer(modifier = Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .background(color, shape = CircleShape)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(text = confidence, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = color)
            }
        }
    }
}