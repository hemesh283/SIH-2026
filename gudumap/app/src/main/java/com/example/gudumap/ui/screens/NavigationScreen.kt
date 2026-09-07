package com.example.gudumap.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
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
import androidx.compose.material3.OutlinedButton
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
import com.example.gudumap.navigation.NavigationState
import com.example.gudumap.ui.components.MapView
import com.example.gudumap.viewmodel.NavigationViewModel
import java.util.Locale

/**
 * Plain-language confidence tier derived from the EKF's numeric uncertainty radius (§19/§23).
 * Thresholds are a first-pass judgment call for demo/UI purposes only -- not validated against
 * real measured accuracy data.
 */
private fun confidenceLevel(radiusMeters: Double): String = when {
    radiusMeters < 5.0 -> "High"
    radiusMeters <= 15.0 -> "Medium"
    else -> "Low"
}

// Solid-fill colors, not a light tint behind same-hue text (§23: a tint background + full-
// saturation text of the same color measured only ~2.8-4.0:1 contrast, failing WCAG AA 4.5:1).
// These are used as solid chip fills with white text, verified >=4.8:1 at any size.
private fun confidenceColor(level: String): Color = when (level) {
    "High" -> Color(0xFF047857)
    "Medium" -> Color(0xFFB45309)
    else -> Color(0xFFDC2626)
}

private fun confidenceCaption(level: String): String = when (level) {
    "High" -> "Position is well-established"
    "Medium" -> "Position may drift slightly"
    else -> "Recalculating -- treat position as approximate"
}

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
            navViewModel.retryLocationUpdatesIfNeeded()
        }
    }

    // Pause/resume sensors+location on app background/foreground (§27), and re-check permission
    // on resume in case it was granted via system Settings while backgrounded.
    val lifecycleOwner = LocalLifecycleOwner.current
    val latestViewModel by rememberUpdatedState(navViewModel)
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> {
                    latestViewModel.resumeNavigation()
                    permissionGranted = ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.ACCESS_FINE_LOCATION
                    ) == PackageManager.PERMISSION_GRANTED
                    latestViewModel.retryLocationUpdatesIfNeeded()
                }
                Lifecycle.Event.ON_PAUSE, Lifecycle.Event.ON_STOP -> {
                    latestViewModel.pauseNavigation()
                }
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    var isMapExpanded by remember { mutableStateOf(false) }
    var blackoutControlStage by remember { mutableStateOf(0) } // 0: GNSS AVAILABLE, 1: START GNSS BLACKOUT
    var showTechnicalDetails by remember { mutableStateOf(false) }

    if (isMapExpanded) {
        // Full-screen map -- MapView's own floating "shrink" button (isExpanded=true shows "↙")
        // is what collapses this back; no other chrome needed here (§29's MapView.kt is
        // otherwise unmodified, invoked with the exact same params as the non-expanded case).
        Box(modifier = Modifier.fillMaxSize()) {
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
                uncertaintyRadiusMeters = navState.uncertaintyRadiusMeters,
                isExpanded = true,
                onToggleExpand = { isMapExpanded = false },
                modifier = Modifier.fillMaxSize()
            )
        }
        return
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8F9FA))
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {

        // ========================================================
        // 1. STATUS BANNER -- plain language, calm color per state
        // ========================================================
        StatusBanner(
            blackoutMode = navState.blackoutMode,
            gnssNavigationMode = navState.gnssNavigationMode
        )

        Spacer(modifier = Modifier.height(12.dp))

        // ========================================================
        // 2. MAP -- MapView.kt itself untouched (§29); same params as the
        // expanded branch above, isExpanded=false here.
        // ========================================================
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(14.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(6.dp)) {
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
                    uncertaintyRadiusMeters = navState.uncertaintyRadiusMeters,
                    isExpanded = false,
                    onToggleExpand = { isMapExpanded = true }
                )
            }
        }

        // ========================================================
        // 3. POSITION CONFIDENCE -- only meaningful once blackout has
        // introduced real drift uncertainty (matches when the numeric
        // Confidence Radius figure in the technical details is meaningful).
        // ========================================================
        if (navState.blackoutMode) {
            Spacer(modifier = Modifier.height(12.dp))
            PlainConfidenceCard(uncertaintyRadiusMeters = navState.uncertaintyRadiusMeters)
        }

        // ========================================================
        // 4. MOTION MODE BADGE -- NEW. Has never been visible in any prior
        // UI version; directly surfaces the pedestrian-safety fallback
        // (§24/§25). Only meaningful once a classification has actually
        // been made (decided once at blackout entry) -- outside blackout
        // motionMode just sits at its neutral "VEHICLE_MODE" default, so
        // showing it then would be misleading, not informative.
        // ========================================================
        if (navState.blackoutMode) {
            Spacer(modifier = Modifier.height(10.dp))
            MotionModeBadge(motionMode = navState.motionMode)
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ========================================================
        // 5. GNSS BLACKOUT TOGGLE -- primary action, stays prominent.
        // ========================================================
        BlackoutControlButton(
            navState = navState,
            blackoutControlStage = blackoutControlStage,
            onArm = { blackoutControlStage = 1 },
            onStart = {
                navViewModel.setBlackoutMode(true)
                blackoutControlStage = 0
            },
            onEnd = {
                navViewModel.setBlackoutMode(false)
                blackoutControlStage = 0
            }
        )

        Spacer(modifier = Modifier.height(14.dp))

        // ========================================================
        // 6. TECHNICAL DETAILS -- collapsed by default.
        // ========================================================
        OutlinedButton(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(10.dp),
            onClick = { showTechnicalDetails = !showTechnicalDetails }
        ) {
            Text(
                text = if (showTechnicalDetails) "Hide technical details" else "Show technical details",
                fontWeight = FontWeight.SemiBold
            )
        }

        AnimatedVisibility(
            visible = showTechnicalDetails,
            enter = expandVertically() + fadeIn(),
            exit = shrinkVertically() + fadeOut()
        ) {
            Column {
                Spacer(modifier = Modifier.height(12.dp))

                // DR Distance / Max Error / ML Latency / Confidence Radius / Heading Conf.
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Color.White),
                    shape = RoundedCornerShape(14.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "BLACKOUT METRICS",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF475569),
                            letterSpacing = 1.sp
                        )
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
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Full Navigation Status list, including Internet -- real backend state
                // since §26/§27's isInternetAvailable merge.
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
                            color = Color(0xFF475569),
                            letterSpacing = 1.sp
                        )
                        Spacer(modifier = Modifier.height(10.dp))

                        val gnssDisplay = when {
                            navState.blackoutMode -> "BLACKOUT"
                            navState.gnssNavigationMode == "GNSS_RECOVERY" -> "RECOVERING"
                            else -> navState.gnssStatus
                        }
                        val navDisplay = if (navState.blackoutMode) "DR" else "GNSS"
                        val gateDisplay = "${navState.latestGateAction} (A:${navState.acceptedCount} C:${navState.clampedCount} R:${navState.rejectedCount})"

                        StatusRow(label = "GNSS", value = gnssDisplay, isGood = gnssDisplay == "AVAILABLE")
                        StatusRow(label = "Navigation", value = navDisplay, isGood = navDisplay == "GNSS")
                        StatusRow(label = "Motion", value = navState.motionState, isGood = true)
                        StatusRow(label = "ML Gate", value = gateDisplay, isGood = navState.latestGateAction == "ACCEPTED")
                        StatusRow(label = "ML", value = navState.mlStatus, isGood = navState.mlStatus == "ACTIVE")
                        StatusRow(label = "EKF", value = navState.ekfStatus, isGood = navState.ekfStatus == "ACTIVE")
                        StatusRow(label = "MAP", value = navState.mapStatus, isGood = true)
                        StatusRow(label = "OFFLINE MAP", value = navState.offlineMapStatus, isGood = navState.offlineMapStatus == "AVAILABLE")
                        StatusRow(label = "Internet", value = if (navState.isInternetAvailable) "AVAILABLE" else "UNAVAILABLE", isGood = navState.isInternetAvailable)

                        if (navState.currentRoadName.isNotBlank()) {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Road: ${navState.currentRoadName}",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Color(0xFF2563EB)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Position
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
                            color = Color(0xFF475569),
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

                // Navigation Metrics
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
                            color = Color(0xFF475569),
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

                // Sensor Status
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
                            color = Color(0xFF475569),
                            letterSpacing = 1.sp
                        )
                        Spacer(modifier = Modifier.height(10.dp))

                        SensorStatusRow(name = "Accelerometer", isActive = navState.accelerometerActive)
                        SensorStatusRow(name = "Gyroscope", isActive = navState.gyroscopeActive)
                        SensorStatusRow(name = "Magnetometer", isActive = navState.magnetometerActive)
                    }
                }
            }
        }

        // Location permission request -- a control, not a status readout, so it stays
        // visible regardless of the technical-details toggle.
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

/**
 * Plain-language status banner (§30 redesign). Three real states, three calm colors -- normal
 * blackout/dead-reckoning operation is the system doing its job, not an error, so it does NOT
 * get alarming red (a prior design flaw); strong red is reserved for a genuine problem state,
 * of which none currently exists as a distinct boolean in NavigationState, so all three states
 * here use calm, non-alarming colors.
 */
@Composable
private fun StatusBanner(blackoutMode: Boolean, gnssNavigationMode: String, modifier: Modifier = Modifier) {
    val (bgColor, textColor, message) = when {
        blackoutMode -> Triple(Color(0xFFEFF6FF), Color(0xFF1D4ED8), "Navigating without GPS")
        gnssNavigationMode == "GNSS_RECOVERY" -> Triple(Color(0xFFFFFBEB), Color(0xFFD97706), "Reconnecting…")
        else -> Triple(Color(0xFFF0FDF4), Color(0xFF16A34A), "Live Tracking")
    }
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = bgColor),
        shape = RoundedCornerShape(14.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 18.dp, horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = message,
                fontSize = 20.sp,
                fontWeight = FontWeight.ExtraBold,
                color = textColor
            )
        }
    }
}

/**
 * Plain-language confidence tier, shown only during blackout. Solid indicator + one-line
 * caption (§23 redesign) rather than a low-contrast tinted pill.
 */
@Composable
private fun PlainConfidenceCard(uncertaintyRadiusMeters: Double, modifier: Modifier = Modifier) {
    val level = confidenceLevel(uncertaintyRadiusMeters)
    val color = confidenceColor(level)
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.06f)),
        shape = RoundedCornerShape(14.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 14.dp, horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .background(color, shape = CircleShape)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Position confidence",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFF475569)
                )
                Text(
                    text = confidenceCaption(level),
                    fontSize = 12.sp,
                    color = Color(0xFF64748B)
                )
            }
            Spacer(modifier = Modifier.width(10.dp))
            Box(
                modifier = Modifier
                    .background(color, shape = RoundedCornerShape(8.dp))
                    .padding(horizontal = 14.dp, vertical = 8.dp)
            ) {
                Text(
                    text = level,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = Color.White
                )
            }
        }
    }
}

/**
 * Motion Mode indicator (§30, NEW) -- the pedestrian-safety fallback (§24/§25) has never been
 * visible in any prior UI version. A small solid-fill chip, not a full-width card: a supporting
 * detail alongside the confidence card, not a headline. Blue/car for VEHICLE_MODE (ML+NHC
 * active, validated domain), amber/walking for CONSERVATIVE_MODE (ML+NHC skipped, bounded
 * fallback displacement only -- §25) -- amber reused deliberately from the same verified-
 * contrast palette as confidenceColor()'s "Medium" tier, since both represent a degraded-but-
 * safe state, not an error.
 */
@Composable
private fun MotionModeBadge(motionMode: String, modifier: Modifier = Modifier) {
    val isConservative = motionMode == "CONSERVATIVE_MODE"
    val icon = if (isConservative) "🚶" else "🚗"
    val label = if (isConservative) "Conservative Mode" else "Vehicle Mode"
    val color = if (isConservative) Color(0xFFB45309) else Color(0xFF2563EB)

    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .background(color, shape = RoundedCornerShape(20.dp))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(text = icon, fontSize = 13.sp)
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = label,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
        }
    }
}

/**
 * The GNSS-blackout trigger/status control -- a control, not a status readout, so it stays
 * visible regardless of the technical-details toggle. Two-stage arm-then-confirm to start
 * (prevents an accidental tap from starting a demo-critical mode), unchanged from prior
 * sessions.
 */
@Composable
private fun BlackoutControlButton(
    navState: NavigationState,
    blackoutControlStage: Int,
    onArm: () -> Unit,
    onStart: () -> Unit,
    onEnd: () -> Unit
) {
    when {
        !navState.hasGpsFix -> {
            // Never allow blackout entry without a real GPS fix -- surface this plainly
            // instead of silently falling back to any default position (Fix 2, §11).
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF64748B)),
                shape = RoundedCornerShape(10.dp),
                enabled = false,
                onClick = {}
            ) {
                Text(text = "WAITING FOR GPS FIX...", fontWeight = FontWeight.Bold)
            }
        }
        navState.blackoutMode -> {
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626)),
                shape = RoundedCornerShape(10.dp),
                onClick = onEnd
            ) {
                Text(text = "GNSS BLACKOUT ACTIVE (TAP TO END)", fontWeight = FontWeight.Bold)
            }
        }
        navState.gnssNavigationMode == "GNSS_RECOVERY" -> {
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706)),
                shape = RoundedCornerShape(10.dp),
                onClick = {}
            ) {
                Text(text = "RECOVERING GNSS...", fontWeight = FontWeight.Bold)
            }
        }
        blackoutControlStage == 1 -> {
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626)),
                shape = RoundedCornerShape(10.dp),
                onClick = onStart
            ) {
                Text(text = "START GNSS BLACKOUT", fontWeight = FontWeight.Bold)
            }
        }
        else -> {
            Button(
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF16A34A)),
                shape = RoundedCornerShape(10.dp),
                onClick = onArm
            ) {
                Text(text = "GNSS AVAILABLE", fontWeight = FontWeight.Bold)
            }
        }
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
            Text(text = title, fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF475569))
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = value, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color(0xFF0F172A))
        }
    }
}

/**
 * Surfaces Android's own magnetometer/rotation-vector reliability signal for heading, so a
 * degraded compass (common inside a vehicle chassis or a tunnel/parking-garage's rebar) is
 * visible rather than silently assumed fine.
 */
@Composable
private fun HeadingConfidenceTile(confidence: String, modifier: Modifier = Modifier) {
    val color = when (confidence) {
        "HIGH" -> Color(0xFF047857)
        "MEDIUM" -> Color(0xFFB45309)
        else -> Color(0xFFB91C1C) // LOW or UNRELIABLE
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
            Text(text = "Heading Conf.", fontSize = 10.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF475569))
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
